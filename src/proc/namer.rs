use crate::{arena::Handle, FastHashMap};
use std::collections::hash_map::Entry;

pub type EntryPointIndex = u16;

#[derive(Debug, Eq, Hash, PartialEq)]
pub enum NameKey {
    GlobalVariable(Handle<crate::GlobalVariable>),
    Type(Handle<crate::Type>),
    StructMember(Handle<crate::Type>, u32),
    Function(Handle<crate::Function>),
    FunctionParameter(Handle<crate::Function>, u32),
    FunctionLocal(Handle<crate::Function>, Handle<crate::LocalVariable>),
    EntryPoint(EntryPointIndex),
    EntryPointLocal(EntryPointIndex, Handle<crate::LocalVariable>),
}

/// This processor assigns names to all the things in a module
/// that may need identifiers in a textual backend.
pub struct Namer {
    unique: FastHashMap<String, u32>,
}

const MAX_PARAM_COUNT: u32 = 100;
const PARAM_NAME: &str = "param";

impl Namer {
    fn sanitize(string: &str) -> String {
        let mut base = string
            .chars()
            .skip_while(|c| c.is_numeric())
            .filter(|&c| c.is_ascii_alphanumeric() || c == '_')
            .collect::<String>();
        // close the name by '_' if the re is a number, so that
        // we can have our own number!
        match base.chars().next_back() {
            Some(c) if !c.is_numeric() => {}
            _ => base.push('_'),
        };
        base
    }

    fn call(&mut self, label_raw: &str) -> String {
        let base = Self::sanitize(label_raw);
        match self.unique.entry(base) {
            Entry::Occupied(mut e) => {
                *e.get_mut() += 1;
                format!("{}{}", e.key(), e.get())
            }
            Entry::Vacant(e) => {
                let name = e.key().to_string();
                e.insert(0);
                name
            }
        }
    }

    fn call_or(&mut self, label: &Option<String>, fallback: &str) -> String {
        self.call(match *label {
            Some(ref name) => name,
            None => fallback,
        })
    }

    pub fn process(
        module: &crate::Module,
        reserved: &[&str],
        output: &mut FastHashMap<NameKey, String>,
    ) {
        let mut this = Namer {
            unique: reserved
                .iter()
                .map(|string| (string.to_string(), 0))
                .collect(),
        };
        // Force the starting suffix for the function parameter names
        // to be higher than whatever we assign explicitly.
        this.unique.insert(PARAM_NAME.to_string(), MAX_PARAM_COUNT);

        for (handle, var) in module.global_variables.iter() {
            let name = this.call_or(&var.name, "global");
            output.insert(NameKey::GlobalVariable(handle), name);
        }

        for (ty_handle, ty) in module.types.iter() {
            let ty_name = this.call_or(&ty.name, "type");
            output.insert(NameKey::Type(ty_handle), ty_name);

            if let crate::TypeInner::Struct { ref members } = ty.inner {
                for (index, member) in members.iter().enumerate() {
                    let name = this.call_or(&member.name, "member");
                    output.insert(NameKey::StructMember(ty_handle, index as u32), name);
                }
            }
        }

        for (fun_handle, fun) in module.functions.iter() {
            let fun_name = this.call_or(&fun.name, "function");
            output.insert(NameKey::Function(fun_handle), fun_name);
            if fun.parameter_types.len() >= MAX_PARAM_COUNT as usize {
                unreachable!()
            }
            for (index, _) in fun.parameter_types.iter().enumerate() {
                let name = format!("{}{}", PARAM_NAME, index + 1);
                output.insert(NameKey::FunctionParameter(fun_handle, index as u32), name);
            }
            for (handle, var) in fun.local_variables.iter() {
                let name = this.call_or(&var.name, "local");
                output.insert(NameKey::FunctionLocal(fun_handle, handle), name);
            }
        }

        for (ep_index, (&(_, ref base_name), ep)) in module.entry_points.iter().enumerate() {
            let ep_name = this.call(base_name);
            output.insert(NameKey::EntryPoint(ep_index as _), ep_name);
            for (handle, var) in ep.function.local_variables.iter() {
                let name = this.call_or(&var.name, "local");
                output.insert(NameKey::EntryPointLocal(ep_index as _, handle), name);
            }
        }
    }
}
