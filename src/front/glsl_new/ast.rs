use crate::{
    Arena, Binding, Constant, Expression, FastHashMap, Function, GlobalVariable, Handle,
    Interpolation, LocalVariable, ShaderStage, Statement, StorageClass, Type,
};

#[derive(Debug)]
pub struct Program {
    pub version: u16,
    pub profile: Profile,
    pub shader_stage: ShaderStage,
    pub lookup_function: FastHashMap<String, Handle<Function>>,
    pub functions: Arena<Function>,
    pub lookup_type: FastHashMap<String, Handle<Type>>,
    pub types: Arena<Type>,
    pub constants: Arena<Constant>,
    pub global_variables: Arena<GlobalVariable>,
    pub lookup_global_variables: FastHashMap<String, Handle<GlobalVariable>>,
    pub context: Context,
}

impl Program {
    pub fn new(shader_stage: ShaderStage) -> Program {
        Program {
            version: 0,
            profile: Profile::Core,
            shader_stage,
            lookup_function: FastHashMap::default(),
            functions: Arena::<Function>::new(),
            lookup_type: FastHashMap::default(),
            types: Arena::<Type>::new(),
            constants: Arena::<Constant>::new(),
            global_variables: Arena::<GlobalVariable>::new(),
            lookup_global_variables: FastHashMap::default(),
            context: Context {
                expressions: Arena::<Expression>::new(),
                local_variables: Arena::<LocalVariable>::new(),
                scopes: vec![FastHashMap::default()],
            },
        }
    }
}

#[derive(Debug)]
pub enum Profile {
    Core,
}

#[derive(Debug)]
pub struct Context {
    pub expressions: Arena<Expression>,
    pub local_variables: Arena<LocalVariable>,
    //TODO: Find less allocation heavy representation
    pub scopes: Vec<FastHashMap<String, Handle<LocalVariable>>>,
}

impl Context {
    pub fn lookup_local_var(&self, name: &str) -> Option<Handle<LocalVariable>> {
        for scope in self.scopes.iter().rev() {
            if let Some(var) = scope.get(name) {
                return Some(*var);
            }
        }
        None
    }

    #[cfg(feature = "glsl-validate")]
    pub fn lookup_local_var_current_scope(&self, name: &str) -> Option<Handle<LocalVariable>> {
        if let Some(current) = self.scopes.last() {
            current.get(name).cloned()
        } else {
            None
        }
    }

    pub fn clear_scopes(&mut self) {
        self.scopes.clear();
        self.scopes.push(FastHashMap::default());
    }

    /// Add variable to current scope
    pub fn add_local_var(&mut self, name: String, handle: Handle<LocalVariable>) {
        if let Some(current) = self.scopes.last_mut() {
            (*current).insert(name, handle);
        }
    }

    /// Add new empty scope
    pub fn push_scope(&mut self) {
        self.scopes.push(FastHashMap::default());
    }

    pub fn remove_current_scope(&mut self) {
        self.scopes.pop();
    }
}

#[derive(Debug)]
pub struct ExpressionRule {
    pub expression: Handle<Expression>,
    pub statements: Vec<Statement>,
}

#[derive(Debug)]
pub enum TypeQualifier {
    StorageClass(StorageClass),
    Binding(Binding),
    Interpolation(Interpolation),
}

#[derive(Debug)]
pub struct VarDeclaration {
    pub type_qualifiers: Vec<TypeQualifier>,
    pub ids_initializers: Vec<(String, Option<ExpressionRule>)>,
    pub ty: Handle<Type>,
}

#[derive(Debug)]
pub enum FunctionCallKind {
    TypeConstructor(Handle<Type>),
    Function(Handle<Expression>),
}

#[derive(Debug)]
pub struct FunctionCall {
    pub kind: FunctionCallKind,
    pub args: Vec<Handle<Expression>>,
    pub statements: Vec<Statement>,
}
