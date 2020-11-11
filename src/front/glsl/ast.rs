use super::error::ErrorKind;
use crate::{
    proc::{ResolveContext, Typifier},
    Arena, BinaryOperator, Binding, Expression, FastHashMap, Function, GlobalVariable, Handle,
    Interpolation, LocalVariable, Module, ShaderStage, Statement, StorageClass, Type,
};

#[derive(Debug)]
pub struct Program {
    pub version: u16,
    pub profile: Profile,
    pub shader_stage: ShaderStage,
    pub entry: Option<String>,
    pub lookup_function: FastHashMap<String, Handle<Function>>,
    pub lookup_type: FastHashMap<String, Handle<Type>>,
    pub lookup_global_variables: FastHashMap<String, Handle<GlobalVariable>>,
    pub context: Context,
    pub module: Module,
}

impl Program {
    pub fn new(shader_stage: ShaderStage, entry: &str) -> Program {
        Program {
            version: 0,
            profile: Profile::Core,
            shader_stage,
            entry: Some(entry.to_string()),
            lookup_function: FastHashMap::default(),
            lookup_type: FastHashMap::default(),
            lookup_global_variables: FastHashMap::default(),
            context: Context {
                expressions: Arena::<Expression>::new(),
                local_variables: Arena::<LocalVariable>::new(),
                scopes: vec![FastHashMap::default()],
                lookup_global_var_exps: FastHashMap::default(),
                typifier: Typifier::new(),
            },
            module: Module::generate_empty(),
        }
    }

    pub fn binary_expr(
        &mut self,
        op: BinaryOperator,
        left: &ExpressionRule,
        right: &ExpressionRule,
    ) -> ExpressionRule {
        ExpressionRule::from_expression(self.context.expressions.append(Expression::Binary {
            op,
            left: left.expression,
            right: right.expression,
        }))
    }

    pub fn resolve_type(
        &mut self,
        handle: Handle<crate::Expression>,
    ) -> Result<&crate::TypeInner, ErrorKind> {
        let functions = Arena::new(); //TODO
        let arguments = Vec::new(); //TODO
        let resolve_ctx = ResolveContext {
            constants: &self.module.constants,
            global_vars: &self.module.global_variables,
            local_vars: &self.context.local_variables,
            functions: &functions,
            arguments: &arguments,
        };
        match self.context.typifier.grow(
            handle,
            &self.context.expressions,
            &mut self.module.types,
            &resolve_ctx,
        ) {
            //TODO: better error report
            Err(_) => Err(ErrorKind::SemanticError("Can't resolve type")),
            Ok(()) => Ok(self.context.typifier.get(handle, &self.module.types)),
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
    pub scopes: Vec<FastHashMap<String, Handle<Expression>>>,
    pub lookup_global_var_exps: FastHashMap<String, Handle<Expression>>,
    pub typifier: Typifier,
}

impl Context {
    pub fn lookup_local_var(&self, name: &str) -> Option<Handle<Expression>> {
        for scope in self.scopes.iter().rev() {
            if let Some(var) = scope.get(name) {
                return Some(*var);
            }
        }
        None
    }

    #[cfg(feature = "glsl-validate")]
    pub fn lookup_local_var_current_scope(&self, name: &str) -> Option<Handle<Expression>> {
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
    pub fn add_local_var(&mut self, name: String, handle: Handle<Expression>) {
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
    pub sampler: Option<Handle<Expression>>,
}

impl ExpressionRule {
    pub fn from_expression(expression: Handle<Expression>) -> ExpressionRule {
        ExpressionRule {
            expression,
            statements: vec![],
            sampler: None,
        }
    }
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
    pub ids_initializers: Vec<(Option<String>, Option<ExpressionRule>)>,
    pub ty: Handle<Type>,
}

#[derive(Debug)]
pub enum FunctionCallKind {
    TypeConstructor(Handle<Type>),
    Function(String),
}

#[derive(Debug)]
pub struct FunctionCall {
    pub kind: FunctionCallKind,
    pub args: Vec<ExpressionRule>,
}
