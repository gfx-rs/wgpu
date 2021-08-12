use hal::CommandEncoder as _;

#[cfg(feature = "trace")]
use core::borrow::Borrow;
use core::iter;
#[cfg(feature = "trace")]
use crate::device::trace::Command as TraceCommand;
use crate::{
    command::{CommandBuffer, CommandEncoderError},
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Token},
    id::{self, Dummy, IdGuard},
    memory_init_tracker::{MemoryInitKind, MemoryInitTrackerAction},
    resource::QuerySet,
    track::UseExtendError2,
    FastHashMap,
};
use thiserror::Error;
use wgt::BufferAddress;

#[derive(Debug)]
pub(super) struct QueryResetMap<'a, A: HalApi + 'a> {
    map: FastHashMap</*Index*/IdGuard<'a, A, QuerySet<Dummy>>, /*(Vec<bool>, Epoch)*/Vec<bool>>,
    // _phantom: PhantomData<A>,
}
impl<'a, A: HalApi> QueryResetMap<'a, A> {
    pub fn new() -> Self {
        Self {
            map: FastHashMap::default(),
            // _phantom: PhantomData,
        }
    }

    pub fn use_query_set(
        &mut self,
        // id: id::QuerySetId,
        query_set: IdGuard<'a, A, QuerySet<Dummy>>,
        query: u32,
    ) -> bool {
        // let (index, epoch, _) = id.unzip();
        let vec_pair = self
            .map
            .entry(/*index*/query_set)
            .or_insert_with(|| (vec![false; query_set.desc.count as usize]/*, epoch*/));

        std::mem::replace(&mut vec_pair/*.0*/[query as usize], true)
    }

    pub fn reset_queries(
        self,
        raw_encoder: &mut A::CommandEncoder,
        // query_set_storage: &Storage<QuerySet<A>, id::QuerySetId>,
        /*backend: wgt::Backend,*/
    ) -> Result<(), id::QuerySetId> where A: HalApi {
        for (/*query_set_id*/query_set, /*(state, epoch)*/state) in self.map.into_iter() {
            /* let id = Id::zip(query_set_id, epoch, /*backend*/A::VARIANT);
            let query_set = query_set_storage.get(id).map_err(|_| id)?; */

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
                        unsafe { raw_encoder.reset_queries(&query_set.raw, start..idx as u32) };
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
    Occlusion,
    Timestamp,
    PipelineStatistics,
}
impl From<wgt::QueryType> for SimplifiedQueryType {
    fn from(q: wgt::QueryType) -> Self {
        match q {
            wgt::QueryType::Occlusion => SimplifiedQueryType::Occlusion,
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

impl<'a, A: HalApi> IdGuard<'a, A, QuerySet<Dummy>> {
    fn validate_query<'b>(
        &'b self,
        // query_set_id: id::QuerySetId,
        query_type: SimplifiedQueryType,
        query_index: u32,
        reset_state: Option<&'b mut QueryResetMap<'a, A>>,
    ) -> Result<&'b A::QuerySet, QueryUseError> {
        // We need to defer our resets because we are in a renderpass, add the usage to the reset map.
        if let Some(reset) = reset_state {
            let used = reset.use_query_set(/*query_set_id*/ *self, query_index);
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

        Ok(&self.raw)
    }

    pub(super) fn validate_and_write_timestamp<'b>(
        self,
        raw_encoder: &mut A::CommandEncoder,
        // query_set_id: id::QuerySetId,
        query_index: u32,
        reset_state: Option<&'b mut QueryResetMap<'a, A>>,
    ) -> Result<(), QueryUseError> {
        let needs_reset = reset_state.is_none();
        let query_set = self.validate_query(
            // query_set_id,
            SimplifiedQueryType::Timestamp,
            query_index,
            reset_state,
        )?;

        unsafe {
            // If we don't have a reset state tracker which can defer resets, we must reset now.
            if needs_reset {
                raw_encoder.reset_queries(&self.raw, query_index..(query_index + 1));
            }
            raw_encoder.write_timestamp(query_set, query_index);
        }

        Ok(())
    }

    pub(super) fn validate_and_begin_pipeline_statistics_query<'b>(
        self,
        raw_encoder: &mut A::CommandEncoder,
        // query_set_id: id::QuerySetId,
        query_index: u32,
        reset_state: Option<&'b mut QueryResetMap<'a, A>>,
        active_query: &'b mut Option<(/*id::QuerySetId*/Self, u32)>,
    ) -> Result<(), QueryUseError> {
        let needs_reset = reset_state.is_none();
        let query_set = self.validate_query(
            // query_set_id,
            SimplifiedQueryType::PipelineStatistics,
            query_index,
            reset_state,
        )?;

        if let Some((_old_id, old_idx)) = active_query.replace((/*query_set_id*/self, query_index)) {
            return Err(QueryUseError::AlreadyStarted {
                active_query_index: old_idx,
                new_query_index: query_index,
            });
        }

        unsafe {
            // If we don't have a reset state tracker which can defer resets, we must reset now.
            if needs_reset {
                raw_encoder.reset_queries(&self.raw, query_index..(query_index + 1));
            }
            raw_encoder.begin_query(query_set, query_index);
        }

        Ok(())
    }
}

pub(super) fn end_pipeline_statistics_query<'a, A: HalApi>(
    raw_encoder: &mut A::CommandEncoder,
    // storage: &Storage<QuerySet<A>, id::QuerySetId>,
    active_query: &mut Option<(/*id::QuerySetId*/IdGuard<'a, A, QuerySet<Dummy>>, u32)>,
) -> Result<(), QueryUseError> {
    if let Some((query_set_id, query_index)) = active_query.take() {
        // We can unwrap here as the validity was validated when the active query was set
        // let query_set = storage.get(query_set_id).unwrap();

        unsafe { raw_encoder.end_query(&query_set_id.raw, query_index) };

        Ok(())
    } else {
        Err(QueryUseError::AlreadyStopped)
    }
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn command_encoder_write_timestamp<A: HalApi>(
        &self,
        command_encoder_id: id::CommandEncoderId,
        query_set_id: &id::QuerySetId,
        query_index: u32,
    ) -> Result<(), QueryError> {
        let hub = A::hub(self);
        let mut token = Token::root();

        let (mut cmd_buf_guard, _) = hub.command_buffers.write(&mut token);
        // let (query_set_guard, _) = hub.query_sets.read(&mut token);

        let cmd_buf = CommandBuffer::get_encoder_mut(&mut cmd_buf_guard, command_encoder_id)?;
        let raw_encoder = cmd_buf.encoder.open();

        let query_set_id = id::expect_backend(query_set_id);
        // FIXME: if bind_group.device() != cmd_buf.device() {
        //   return Err(RenderCommandError::InvalidBindGroup(bind_group_id.clone()))
        //          .map_pass_err(scope);
        // }

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
            let query_set_id = id::QuerySetId::as_usize::<A>(query_set_id.borrow());
            list.push(TraceCommand::WriteTimestamp {
                query_set_id,
                query_index,
            });
        }

        let query_set = cmd_buf
            .trackers
            .query_sets
            .use_extend(/*&*query_set_guard, */query_set_id, (), ())
            .unwrap_or_else(|UseExtendError2::Conflict(err)| match err {})
            /*.map_err(|e| match e {
                UseExtendError::InvalidResource => QueryError::InvalidQuerySet(query_set_id),
                _ => unreachable!(),
            })?*/;

        query_set.validate_and_write_timestamp(raw_encoder, /*query_set_id, */query_index, None)?;

        Ok(())
    }

    pub fn command_encoder_resolve_query_set<A: HalApi>(
        &self,
        command_encoder_id: id::CommandEncoderId,
        query_set_id: &id::QuerySetId,
        start_query: u32,
        query_count: u32,
        destination: id::BufferId,
        destination_offset: BufferAddress,
    ) -> Result<(), QueryError> {
        let hub = A::hub(self);
        let mut token = Token::root();

        let (buffer_guard, mut token) = hub.buffers.read(&mut token);
        let (mut cmd_buf_guard, _) = hub.command_buffers.write(&mut token);
        // let (query_set_guard, _) = hub.query_sets.read(&mut token);

        let cmd_buf = CommandBuffer::get_encoder_mut(&mut cmd_buf_guard, command_encoder_id)?;
        let raw_encoder = cmd_buf.encoder.open();

        let query_set_id = id::expect_backend(query_set_id);
        // FIXME: if bind_group.device() != cmd_buf.device() {
        //   return Err(RenderCommandError::InvalidBindGroup(bind_group_id.clone()))
        //          .map_pass_err(scope);
        // }

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
            let query_set_id = id::QuerySetId::as_usize::<A>(query_set_id.borrow());
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
            .use_extend(/*&*query_set_guard, */query_set_id, (), ())
            .unwrap_or_else(|UseExtendError2::Conflict(err)| match err {})
            /* .map_err(|e| match e {
                UseExtendError::InvalidResource => QueryError::InvalidQuerySet(query_set_id),
                _ => unreachable!(),
            })?*/;

        let (dst_buffer, dst_pending) = cmd_buf
            .trackers
            .buffers
            .use_replace(&*buffer_guard, destination, (), hal::BufferUses::COPY_DST)
            .map_err(QueryError::InvalidBuffer)?;
        let dst_barrier = dst_pending.map(|pending| pending.into_hal(dst_buffer));

        if !dst_buffer.usage.contains(wgt::BufferUsages::COPY_DST) {
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
            wgt::QueryType::Occlusion => 1,
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
            raw_encoder.transition_buffers(dst_barrier);
            raw_encoder.copy_query_results(
                &query_set.raw,
                start_query..end_query,
                dst_buffer.raw.as_ref().unwrap(),
                destination_offset,
                wgt::BufferSize::new_unchecked(stride as u64),
            );
        }

        Ok(())
    }
}
