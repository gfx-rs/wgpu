/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use hal::command::CommandBuffer as _;

#[cfg(feature = "trace")]
use crate::device::trace::Command as TraceCommand;
use crate::{
    command::{CommandBuffer, CommandEncoderError},
    device::all_buffer_stages,
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Storage, Token},
    id::{self, Id, TypedId},
    memory_init_tracker::{MemoryInitKind, MemoryInitTrackerAction},
    resource::{BufferUse, QuerySet},
    track::UseExtendError,
    Epoch, FastHashMap, Index,
};
use std::{iter, marker::PhantomData};
use thiserror::Error;
use wgt::BufferAddress;

#[derive(Debug)]
pub(super) struct QueryResetMap<B: hal::Backend> {
    map: FastHashMap<Index, (Vec<bool>, Epoch)>,
    _phantom: PhantomData<B>,
}
impl<B: hal::Backend> QueryResetMap<B> {
    pub fn new() -> Self {
        Self {
            map: FastHashMap::default(),
            _phantom: PhantomData,
        }
    }

    pub fn use_query_set(
        &mut self,
        id: id::QuerySetId,
        query_set: &QuerySet<B>,
        query: u32,
    ) -> bool {
        let (index, epoch, _) = id.unzip();
        let vec_pair = self
            .map
            .entry(index)
            .or_insert_with(|| (vec![false; query_set.desc.count as usize], epoch));

        std::mem::replace(&mut vec_pair.0[query as usize], true)
    }

    pub fn reset_queries(
        self,
        cmd_buf_raw: &mut B::CommandBuffer,
        query_set_storage: &Storage<QuerySet<B>, id::QuerySetId>,
        backend: wgt::Backend,
    ) -> Result<(), id::QuerySetId> {
        for (query_set_id, (state, epoch)) in self.map.into_iter() {
            let id = Id::zip(query_set_id, epoch, backend);
            let query_set = query_set_storage.get(id).map_err(|_| id)?;

            debug_assert_eq!(state.len(), query_set.desc.count as usize);

            // Need to find all "runs" of values which need resets. If the state vector is:
            // [false, true, true, false, true], we want to reset [1..3, 4..5]. This minimizes
            // the amount of resets needed.
            let mut run_start: Option<u32> = None;
            for (idx, value) in state.into_iter().chain(iter::once(false)).enumerate() {
                match (run_start, value) {
                    // We're inside of a run, do nothing
                    (Some(..), true) => {}
                    // We've hit the end of a run, dispatch a reset
                    (Some(start), false) => {
                        run_start = None;
                        unsafe { cmd_buf_raw.reset_query_pool(&query_set.raw, start..idx as u32) };
                    }
                    // We're starting a run
                    (None, true) => {
                        run_start = Some(idx as u32);
                    }
                    // We're in a run of falses, do nothing.
                    (None, false) => {}
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SimplifiedQueryType {
    Timestamp,
    PipelineStatistics,
}
impl From<wgt::QueryType> for SimplifiedQueryType {
    fn from(q: wgt::QueryType) -> Self {
        match q {
            wgt::QueryType::Timestamp => SimplifiedQueryType::Timestamp,
            wgt::QueryType::PipelineStatistics(..) => SimplifiedQueryType::PipelineStatistics,
        }
    }
}

/// Error encountered when dealing with queries
#[derive(Clone, Debug, Error)]
pub enum QueryError {
    #[error(transparent)]
    Encoder(#[from] CommandEncoderError),
    #[error("Error encountered while trying to use queries")]
    Use(#[from] QueryUseError),
    #[error("Error encountered while trying to resolve a query")]
    Resolve(#[from] ResolveError),
    #[error("Buffer {0:?} is invalid or destroyed")]
    InvalidBuffer(id::BufferId),
    #[error("QuerySet {0:?} is invalid or destroyed")]
    InvalidQuerySet(id::QuerySetId),
}

/// Error encountered while trying to use queries
#[derive(Clone, Debug, Error)]
pub enum QueryUseError {
    #[error("Query {query_index} is out of bounds for a query set of size {query_set_size}")]
    OutOfBounds {
        query_index: u32,
        query_set_size: u32,
    },
    #[error("Query {query_index} has already been used within the same renderpass. Queries must only be used once per renderpass")]
    UsedTwiceInsideRenderpass { query_index: u32 },
    #[error("Query {new_query_index} was started while query {active_query_index} was already active. No more than one statistic or occlusion query may be active at once")]
    AlreadyStarted {
        active_query_index: u32,
        new_query_index: u32,
    },
    #[error("Query was stopped while there was no active query")]
    AlreadyStopped,
    #[error("A query of type {query_type:?} was started using a query set of type {set_type:?}")]
    IncompatibleType {
        set_type: SimplifiedQueryType,
        query_type: SimplifiedQueryType,
    },
}

/// Error encountered while trying to resolve a query.
#[derive(Clone, Debug, Error)]
pub enum ResolveError {
    #[error("Queries can only be resolved to buffers that contain the COPY_DST usage")]
    MissingBufferUsage,
    #[error("Resolving queries {start_query}..{end_query} would overrun the query set of size {query_set_size}")]
    QueryOverrun {
        start_query: u32,
        end_query: u32,
        query_set_size: u32,
    },
    #[error("Resolving queries {start_query}..{end_query} ({stride} byte queries) will end up overrunning the bounds of the destination buffer of size {buffer_size} using offsets {buffer_start_offset}..{buffer_end_offset}")]
    BufferOverrun {
        start_query: u32,
        end_query: u32,
        stride: u32,
        buffer_size: BufferAddress,
        buffer_start_offset: BufferAddress,
        buffer_end_offset: BufferAddress,
    },
}

impl<B: GfxBackend> QuerySet<B> {
    fn validate_query(
        &self,
        query_set_id: id::QuerySetId,
        query_type: SimplifiedQueryType,
        query_index: u32,
        reset_state: Option<&mut QueryResetMap<B>>,
    ) -> Result<hal::query::Query<'_, B>, QueryUseError> {
        // We need to defer our resets because we are in a renderpass, add the usage to the reset map.
        if let Some(reset) = reset_state {
            let used = reset.use_query_set(query_set_id, self, query_index);
            if used {
                return Err(QueryUseError::UsedTwiceInsideRenderpass { query_index });
            }
        }

        let simple_set_type = SimplifiedQueryType::from(self.desc.ty);
        if simple_set_type != query_type {
            return Err(QueryUseError::IncompatibleType {
                query_type,
                set_type: simple_set_type,
            });
        }

        if query_index >= self.desc.count {
            return Err(QueryUseError::OutOfBounds {
                query_index,
                query_set_size: self.desc.count,
            });
        }

        let hal_query = hal::query::Query::<B> {
            pool: &self.raw,
            id: query_index,
        };

        Ok(hal_query)
    }

    pub(super) fn validate_and_write_timestamp(
        &self,
        cmd_buf_raw: &mut B::CommandBuffer,
        query_set_id: id::QuerySetId,
        query_index: u32,
        reset_state: Option<&mut QueryResetMap<B>>,
    ) -> Result<(), QueryUseError> {
        let needs_reset = reset_state.is_none();
        let hal_query = self.validate_query(
            query_set_id,
            SimplifiedQueryType::Timestamp,
            query_index,
            reset_state,
        )?;

        unsafe {
            // If we don't have a reset state tracker which can defer resets, we must reset now.
            if needs_reset {
                cmd_buf_raw.reset_query_pool(&self.raw, query_index..(query_index + 1));
            }
            cmd_buf_raw.write_timestamp(hal::pso::PipelineStage::BOTTOM_OF_PIPE, hal_query);
        }

        Ok(())
    }

    pub(super) fn validate_and_begin_pipeline_statistics_query(
        &self,
        cmd_buf_raw: &mut B::CommandBuffer,
        query_set_id: id::QuerySetId,
        query_index: u32,
        reset_state: Option<&mut QueryResetMap<B>>,
        active_query: &mut Option<(id::QuerySetId, u32)>,
    ) -> Result<(), QueryUseError> {
        let needs_reset = reset_state.is_none();
        let hal_query = self.validate_query(
            query_set_id,
            SimplifiedQueryType::PipelineStatistics,
            query_index,
            reset_state,
        )?;

        if let Some((_old_id, old_idx)) = active_query.replace((query_set_id, query_index)) {
            return Err(QueryUseError::AlreadyStarted {
                active_query_index: old_idx,
                new_query_index: query_index,
            });
        }

        unsafe {
            // If we don't have a reset state tracker which can defer resets, we must reset now.
            if needs_reset {
                cmd_buf_raw.reset_query_pool(&self.raw, query_index..(query_index + 1));
            }
            cmd_buf_raw.begin_query(hal_query, hal::query::ControlFlags::empty());
        }

        Ok(())
    }
}

pub(super) fn end_pipeline_statistics_query<B: GfxBackend>(
    cmd_buf_raw: &mut B::CommandBuffer,
    storage: &Storage<QuerySet<B>, id::QuerySetId>,
    active_query: &mut Option<(id::QuerySetId, u32)>,
) -> Result<(), QueryUseError> {
    if let Some((query_set_id, query_index)) = active_query.take() {
        // We can unwrap here as the validity was validated when the active query was set
        let query_set = storage.get(query_set_id).unwrap();

        let hal_query = hal::query::Query::<B> {
            pool: &query_set.raw,
            id: query_index,
        };

        unsafe { cmd_buf_raw.end_query(hal_query) }

        Ok(())
    } else {
        Err(QueryUseError::AlreadyStopped)
    }
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn command_encoder_write_timestamp<B: GfxBackend>(
        &self,
        command_encoder_id: id::CommandEncoderId,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), QueryError> {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut cmd_buf_guard, mut token) = hub.command_buffers.write(&mut token);
        let (query_set_guard, _) = hub.query_sets.read(&mut token);

        let cmd_buf = CommandBuffer::get_encoder_mut(&mut cmd_buf_guard, command_encoder_id)?;
        let cmd_buf_raw = cmd_buf.raw.last_mut().unwrap();

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
            list.push(TraceCommand::WriteTimestamp {
                query_set_id,
                query_index,
            });
        }

        let query_set = cmd_buf
            .trackers
            .query_sets
            .use_extend(&*query_set_guard, query_set_id, (), ())
            .map_err(|e| match e {
                UseExtendError::InvalidResource => QueryError::InvalidQuerySet(query_set_id),
                _ => unreachable!(),
            })?;

        query_set.validate_and_write_timestamp(cmd_buf_raw, query_set_id, query_index, None)?;

        Ok(())
    }

    pub fn command_encoder_resolve_query_set<B: GfxBackend>(
        &self,
        command_encoder_id: id::CommandEncoderId,
        query_set_id: id::QuerySetId,
        start_query: u32,
        query_count: u32,
        destination: id::BufferId,
        destination_offset: BufferAddress,
    ) -> Result<(), QueryError> {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut cmd_buf_guard, mut token) = hub.command_buffers.write(&mut token);
        let (query_set_guard, mut token) = hub.query_sets.read(&mut token);
        let (buffer_guard, _) = hub.buffers.read(&mut token);

        let cmd_buf = CommandBuffer::get_encoder_mut(&mut cmd_buf_guard, command_encoder_id)?;
        let cmd_buf_raw = cmd_buf.raw.last_mut().unwrap();

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
            list.push(TraceCommand::ResolveQuerySet {
                query_set_id,
                start_query,
                query_count,
                destination,
                destination_offset,
            });
        }

        let query_set = cmd_buf
            .trackers
            .query_sets
            .use_extend(&*query_set_guard, query_set_id, (), ())
            .map_err(|e| match e {
                UseExtendError::InvalidResource => QueryError::InvalidQuerySet(query_set_id),
                _ => unreachable!(),
            })?;

        let (dst_buffer, dst_pending) = cmd_buf
            .trackers
            .buffers
            .use_replace(&*buffer_guard, destination, (), BufferUse::COPY_DST)
            .map_err(QueryError::InvalidBuffer)?;
        let dst_barrier = dst_pending.map(|pending| pending.into_hal(dst_buffer));

        if !dst_buffer.usage.contains(wgt::BufferUsage::COPY_DST) {
            return Err(ResolveError::MissingBufferUsage.into());
        }

        let end_query = start_query + query_count;
        if end_query > query_set.desc.count {
            return Err(ResolveError::QueryOverrun {
                start_query,
                end_query,
                query_set_size: query_set.desc.count,
            }
            .into());
        }

        let elements_per_query = match query_set.desc.ty {
            wgt::QueryType::PipelineStatistics(ps) => ps.bits().count_ones(),
            wgt::QueryType::Timestamp => 1,
        };
        let stride = elements_per_query * wgt::QUERY_SIZE;
        let bytes_used = (stride * query_count) as BufferAddress;

        let buffer_start_offset = destination_offset;
        let buffer_end_offset = buffer_start_offset + bytes_used;

        if buffer_end_offset > dst_buffer.size {
            return Err(ResolveError::BufferOverrun {
                start_query,
                end_query,
                stride,
                buffer_size: dst_buffer.size,
                buffer_start_offset,
                buffer_end_offset,
            }
            .into());
        }

        cmd_buf.buffer_memory_init_actions.extend(
            dst_buffer
                .initialization_status
                .check(buffer_start_offset..buffer_end_offset)
                .map(|range| MemoryInitTrackerAction {
                    id: destination,
                    range,
                    kind: MemoryInitKind::ImplicitlyInitialized,
                }),
        );

        unsafe {
            cmd_buf_raw.pipeline_barrier(
                all_buffer_stages()..hal::pso::PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                dst_barrier,
            );
            cmd_buf_raw.copy_query_pool_results(
                &query_set.raw,
                start_query..end_query,
                &dst_buffer.raw.as_ref().unwrap().0,
                destination_offset,
                stride,
                hal::query::ResultFlags::WAIT | hal::query::ResultFlags::BITS_64,
            );
        }

        Ok(())
    }
}
