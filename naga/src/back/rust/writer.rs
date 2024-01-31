use crate::back;
use crate::back::rust::Target;
use crate::proc::{self, NameKey};
use crate::ShaderStage;
use crate::{valid, Binding};
use crate::{
    Arena, BinaryOperator, Constant, Expression, Function, Handle, Literal, LocalVariable,
    MathFunction, Module, Scalar, ScalarKind, Statement, SwizzleComponent, Type, TypeInner,
    UnaryOperator, UniqueArena, VectorSize,
};
use crate::{BuiltIn, Interpolation, Sampling};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::punctuated::Punctuated;
use syn::{
    self, token, Attribute, BinOp, Block as SynBlock, Expr, ExprArray, ExprAssign, ExprCall,
    ExprField, ExprGroup, ExprIf, ExprIndex, ExprLit, ExprMethodCall, ExprParen, ExprPath,
    ExprReturn, ExprUnary, File, FnArg, Generics, Ident, ItemConst, ItemFn, Local, LocalInit,
    Member, Meta, MetaList, Pat, PatIdent, PatType, Path, PathSegment, ReturnType, Signature, Stmt,
    UnOp, Visibility,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WriterError {
    #[error("Missing function name")]
    MissingFunctionName,
    #[error("Missing global variable name")]
    MissingGlobalVariableName,
    #[error("Missing local variable name")]
    MissingLocalVariableName,
    #[error("Missing struct member name")]
    MissingStructMemberName,
    #[error("Unsuppored scalar type ({0:?})")]
    UnsupportedScalarType(Scalar),
    #[error("Unsuppored vector type ({0:?}) of scalar ({1:?})")]
    UnsupportedVectorType(VectorSize, Scalar),
    #[error("Unsuppored matrix type ({0:?}x{1:?}) of scalar ({2:?})")]
    UnsupportedMatrixType(VectorSize, VectorSize, Scalar),
    #[error("Unsuppored binary operator({0:?}")]
    UnsupportedBinaryOperator(BinaryOperator),
    #[error("Unknown type for expression ({0:?}")]
    UnknownExpressionType(Expression),
    #[error("Missing argument for math function ({0:?}")]
    MissingMathFunctionArgument(MathFunction),
    #[error("Mismatched vector arg size: expected ({0:?}) got ({1:?})")]
    MismatchedVectorArgSize(VectorSize, VectorSize),
    #[error("Mismatched vector arg scalar type: expected ({0:?}) got ({1:?})")]
    MismatchedVectorArgScalarType(Scalar, Scalar),
    #[error("Invalid vecoctor index ({1}) for vector of size ({0:?})")]
    InvalidVectorIndex(VectorSize, u32),
    #[error("Invalid matrix index ({0:?}x{1:?}) at position ({2})")]
    InvalidMatrixIndex(VectorSize, VectorSize, u32),
    #[error("Missing constant name")]
    MissingConstantName,
    #[error("Local variable {0:?} found outside of a function")]
    LocalVariableOutsideOfFunction(LocalVariable),
}

pub(crate) fn map_builtin_to_rust_gpu(b: &BuiltIn) -> &'static str {
    use crate::BuiltIn::*;
    match b {
        Position { invariant: _ } => "position",
        ViewIndex => "view_index",
        // vertex
        BaseInstance => "base_instance",
        BaseVertex => "base_vertex",
        ClipDistance => "clip_distance",
        CullDistance => "cull_distance",
        InstanceIndex => "instance_index",
        PointSize => "point_size",
        VertexIndex => "vertex_index",
        // fragment
        FragDepth => "frag_depth",
        PointCoord => "point_coord",
        FrontFacing => "front_facing",
        PrimitiveIndex => "primitive_index",
        SampleIndex => "sample_index",
        SampleMask => "sample_mask",
        // compute
        GlobalInvocationId => "global_invocation_id",
        LocalInvocationId => "local_invocation_id",
        LocalInvocationIndex => "local_invocation_index",
        WorkGroupId => "work_group_id",
        WorkGroupSize => "work_group_size",
        NumWorkGroups => "num_work_groups",
    }
}

pub(crate) fn map_type_to_glam(ty: &Type) -> Result<String, WriterError> {
    use ScalarKind::*;
    use VectorSize::*;
    match ty.inner {
        TypeInner::Scalar(scalar @ Scalar { kind, width }) => {
            let ty = match (kind, width) {
                (Sint, 1) => "i8",
                (Sint, 2) => "i16",
                (Sint, 4) | (AbstractInt, _) => "i32",
                (Sint, 8) => "i64",
                (Uint, 1) => "u8",
                (Uint, 2) => "u16",
                (Uint, 4) => "u32",
                (Uint, 8) => "u64",
                (Float, 4) | (AbstractFloat, _) => "f32",
                (Float, 8) => "f64",
                (Bool, _) => "bool",
                _ => return Err(WriterError::UnsupportedScalarType(scalar)),
            };
            Ok(ty.to_string())
        }
        TypeInner::Vector { size, scalar } => {
            let ty = match (size, scalar.kind, scalar.width) {
                // Floats.
                (Bi, AbstractFloat, ..) => "Vec2",
                (Bi, Float, 2) => "Vec2",
                (Bi, Float, 4) => "Vec2",
                (Bi, Float, 8) => "DVec2",
                (Tri, Float, 2) => "Vec3",
                (Tri, Float, 4) => "Vec3",
                (Tri, Float, 8) => "DVec3",
                (Quad, Float, 2) => "Vec4",
                (Quad, Float, 4) => "Vec4",
                (Quad, Float, 8) => "DVec4",
                // Unsigned ints.
                (Bi, Uint, 2) => "U16Vec2",
                (Bi, Uint, 4) => "UVec2",
                (Bi, Uint, 8) => "U64Vec2",
                (Tri, Uint, 2) => "U16Vec3",
                (Tri, Uint, 4) => "UVec3",
                (Tri, Uint, 8) => "U64DVec3",
                (Quad, Uint, 2) => "U16Vec4",
                (Quad, Uint, 4) => "UVec4",
                (Quad, Uint, 8) => "U64DVec4",
                // Signed ints.
                (Bi, AbstractInt, ..) => "Vec2",
                (Bi, Sint, 2) => "I16Vec2",
                (Bi, Sint, 4) => "IVec2",
                (Bi, Sint, 8) => "I64Vec2",
                (Tri, Sint, 2) => "I16Vec3",
                (Tri, Sint, 4) => "IVec3",
                (Tri, Sint, 8) => "I64DVec3",
                (Quad, Sint, 2) => "I16Vec4",
                (Quad, Sint, 4) => "IVec4",
                (Quad, Sint, 8) => "I64DVec4",
                _ => return Err(WriterError::UnsupportedVectorType(size, scalar)),
            };
            Ok(ty.to_string())
        }
        TypeInner::Matrix {
            columns,
            rows,
            scalar,
        } => {
            let ty = match (columns, rows, scalar.kind, scalar.width) {
                // Floats.
                (Bi, Bi, AbstractFloat, ..) => "Mat2",
                (Bi, Bi, Float, 2) => "Mat2",
                (Bi, Bi, Float, 4) => "Mat2",
                (Bi, Bi, Float, 8) => "DMat2",
                (Tri, Tri, Float, 2) => "Mat3",
                (Tri, Tri, Float, 4) => "Mat3",
                (Tri, Tri, Float, 8) => "DMat3",
                (Quad, Quad, Float, 2) => "Mat4",
                (Quad, Quad, Float, 4) => "Mat4",
                (Quad, Quad, Float, 8) => "DMat4",
                // Unsigned ints.
                (Bi, Bi, Uint, 2) => "U16Mat2",
                (Bi, Bi, Uint, 4) => "UMat2",
                (Bi, Bi, Uint, 8) => "U64Mat2",
                (Tri, Tri, Uint, 2) => "U16Mat3",
                (Tri, Tri, Uint, 4) => "UMat3",
                (Tri, Tri, Uint, 8) => "U64DMat3",
                (Quad, Quad, Uint, 2) => "U16Mat4",
                (Quad, Quad, Uint, 4) => "UMat4",
                (Quad, Quad, Uint, 8) => "U64Mat4",
                // Signed ints.
                (Bi, Bi, AbstractInt, ..) => "IMat2",
                (Bi, Bi, Sint, 2) => "I16Mat2",
                (Bi, Bi, Sint, 4) => "IMat2",
                (Bi, Bi, Sint, 8) => "I64Mat2",
                (Tri, Tri, Sint, 2) => "I16Mat3",
                (Tri, Tri, Sint, 4) => "IMat3",
                (Tri, Tri, Sint, 8) => "I64DMat3",
                (Quad, Quad, Sint, 2) => "I16Mat4",
                (Quad, Quad, Sint, 4) => "IMat4",
                (Quad, Quad, Sint, 8) => "I64Mat4",
                // Bools.
                (Bi, Bi, Bool, ..) => "BMat2",
                (Tri, Tri, Bool, ..) => "BMat3",
                (Quad, Quad, Bool, ..) => "BMat4",
                _ => return Err(WriterError::UnsupportedMatrixType(columns, rows, scalar)),
            };
            Ok(ty.to_string())
        }
        _ => todo!("map type: {ty:?}"),
    }
}

bitflags::bitflags! {
    #[cfg_attr(feature = "serialize", derive(serde::Serialize))]
    #[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct WriterFlags: u32 {
        const EXPLICIT_TYPES = 0x1;
        const INFER_BUILTINS = 0x2;
    }
}

pub struct Writer {
    target: Target,
    flags: WriterFlags,
    names: crate::FastHashMap<NameKey, String>,
    namer: proc::Namer,
    named_expressions: crate::NamedExpressions,
    entrypoint_functions: Vec<Handle<Function>>,
}

impl Writer {
    pub fn new(target: Target, flags: WriterFlags) -> Self {
        Writer {
            flags,
            target,
            names: crate::FastHashMap::default(),
            namer: proc::Namer {
                allow_numeric_end: true,
                ..Default::default()
            },
            named_expressions: crate::NamedExpressions::default(),
            entrypoint_functions: vec![],
        }
    }

    fn reset(&mut self, module: &Module) {
        log::trace!("reset");
        self.names.clear();
        self.namer.reset(
            module,
            crate::keywords::rust::RESERVED,
            &[],
            &[],
            &["_"],
            &mut self.names,
        );
        self.named_expressions.clear();
        self.entrypoint_functions.clear();
    }

    pub fn write_module(
        &mut self,
        module: &Module,
        info: &valid::ModuleInfo,
    ) -> Result<File, WriterError> {
        self.reset(module);

        // Convert all named constants.
        log::trace!("Converting named constants");
        let constants = module
            .constants
            .iter()
            .filter(|&(_, c)| c.name.is_some())
            .map(|(handle, c)| {
                self.convert_global_constant(module, &handle, c)
                    .map(syn::Item::Const)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Convert all entry points.
        log::trace!("Converting entry points");
        let entry_points = module
            .entry_points
            .iter()
            .enumerate()
            .map(|(index, ep)| {
                log::debug!("{:#?}", ep);
                let tokens = match ep.stage {
                    ShaderStage::Vertex => quote!(vertex),
                    ShaderStage::Fragment => quote!(fragment),
                    ShaderStage::Compute => quote!(compute),
                };

                // TODO: support workgroup size for compute.

                let meta = Meta::List(MetaList {
                    path: Path::from(Ident::new("spirv", Span::call_site())),
                    delimiter: syn::MacroDelimiter::Paren(token::Paren::default()),
                    tokens,
                });

                let mut attrs = vec![Attribute {
                    pound_token: Default::default(),
                    style: syn::AttrStyle::Outer,
                    bracket_token: Default::default(),
                    meta,
                }];

                // This is a bit wonky as the frontend creates an unnamed function for
                // each entrypoint which has the inputs as arguments and the outputs as
                // return values. The synthetic function then calls the user's
                // entrypoint function. We don't need to do this with `rust-gpu` as we
                // can just annotate the user's entrypoint function. So we find the call
                // to the user's function and copy the sythetic function's arguments and
                // return functions to it.

                let handle = ep
                    .function
                    .body
                    .into_iter()
                    .filter_map(|x| match x {
                        Statement::Call { function, .. } => Some(function.clone()),
                        _ => None,
                    })
                    .collect::<Vec<Handle<Function>>>()
                    .first()
                    .expect("entrypoint function")
                    .clone();

                self.entrypoint_functions.push(handle);

                let mut function = module.functions[handle].clone();
                function.arguments = ep.function.arguments.clone();
                function.result = ep.function.result.clone();

                let func_ctx = back::FunctionCtx {
                    ty: back::FunctionType::EntryPoint(index as u16),
                    info: info.get_entry_point(index),
                    expressions: &function.expressions,
                    named_expressions: &function.named_expressions,
                };
                let mut f = self.convert_function(module, &handle, &function, &func_ctx)?;

                f.attrs.append(&mut attrs);
                Ok(syn::Item::Fn(f))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Convert regular functions.
        log::trace!("Converting regular functions");
        let funcs = module
            .functions
            .iter()
            .filter_map(|(handle, function)| {
                // Filter out endpoint functions...they are processed above.
                if self.entrypoint_functions.contains(&handle) {
                    return None;
                }
                let fun_info = &info[handle];

                let func_ctx = back::FunctionCtx {
                    ty: back::FunctionType::Function(handle),
                    info: fun_info,
                    expressions: &function.expressions,
                    named_expressions: &function.named_expressions,
                };

                Some(
                    self.convert_function(module, &handle, function, &func_ctx)
                        .map(syn::Item::Fn),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Put everything we have converted together.
        let items: Vec<syn::Item> = constants
            .into_iter()
            .chain(entry_points.into_iter().chain(funcs.into_iter()))
            .collect();

        Ok(File {
            shebang: None,
            attrs: vec![],
            items,
        })
    }

    fn convert_global_constant(
        &mut self,
        module: &Module,
        handle: &Handle<Constant>,
        constant: &Constant,
    ) -> Result<ItemConst, WriterError> {
        log::trace!("Converting global constant");
        // Create an identifier for the name.
        let name = self.names[&NameKey::Constant(*handle)].clone();
        let ident = Ident::new(&name, Span::call_site());

        // Create the type of the constant
        let ty = self.convert_type(&module.types, &constant.ty)?;

        // Get the expression.
        let expr = self.convert_const_expression(&module, &constant.init)?;

        // Construct the const item
        Ok(ItemConst {
            attrs: vec![],
            vis: Visibility::Inherited,
            const_token: Default::default(),
            ident,
            colon_token: Default::default(),
            ty: Box::new(ty),
            eq_token: Default::default(),
            expr: Box::new(expr),
            semi_token: Default::default(),
            generics: Generics::default(),
        })
    }

    fn convert_nonconst_expression(
        &mut self,
        module: &Module,
        function: Option<&Handle<Function>>,
        expressions: &Arena<Expression>,
        expr: &Handle<Expression>,
    ) -> Result<Expr, WriterError> {
        log::trace!("Converting non-const expression");
        let local_variables: Arena<LocalVariable> = if let Some(handle) = function {
            let func = &module.functions[*handle];
            log::trace!("Local variable has function");
            func.local_variables.clone()
        } else {
            log::trace!("Empty local variable arena");
            Arena::new()
        };
        self.convert_possibly_const_expression(
            module,
            function,
            expr,
            &local_variables,
            expressions,
            |writer, expr| writer.convert_nonconst_expression(module, function, expressions, &expr),
        )
    }

    fn convert_const_expression(
        &mut self,
        module: &Module,
        expr: &Handle<Expression>,
    ) -> Result<Expr, WriterError> {
        let local_variables: Arena<LocalVariable> = Arena::new();
        self.convert_possibly_const_expression(
            module,
            None,
            expr,
            &local_variables,
            &module.const_expressions,
            |writer, expr| writer.convert_const_expression(module, expr),
        )
    }

    fn get_expression_type(
        &mut self,
        types: &UniqueArena<Type>,
        local_variables: &Arena<LocalVariable>,
        expressions: &Arena<Expression>,
        expr: &Expression,
    ) -> Result<Type, WriterError> {
        log::trace!("Getting type for expression {expr:?}");
        match expr {
            Expression::Constant(_handle) => {
                log::trace!("Getting type for constant expression");
                todo!()
            }
            Expression::Compose { ty, .. } => {
                log::trace!("Getting type for compose expression");
                Ok(types[*ty].clone())
            }

            Expression::Access { base, .. } => {
                log::trace!("Getting type for access expression");
                // For Access expressions, the type is the type of the base expression.
                self.get_expression_type(types, local_variables, expressions, &expressions[*base])
            }
            Expression::Load { pointer } => {
                // Dereference the pointer to get the expression it refers to.
                self.get_expression_type(
                    types,
                    local_variables,
                    expressions,
                    &expressions[*pointer],
                )
            }
            Expression::GlobalVariable(_handle) => todo!(),
            Expression::LocalVariable(handle) => {
                log::trace!("Getting type for local variable expression");
                let local_var = &local_variables[*handle];
                Ok(types[local_var.ty].clone())
            }
            Expression::Binary { left, .. } | Expression::Unary { expr: left, .. } => {
                log::trace!("Getting type for binary or unary expression");
                // For Binary and Unary expressions, the type is usually the type of the left operand.
                self.get_expression_type(types, local_variables, expressions, &expressions[*left])
            }
            Expression::Literal(literal) => {
                log::trace!("Getting type for literal expression");
                match literal {
                    Literal::AbstractFloat(_) | Literal::F32(_) => Ok(Type {
                        name: Default::default(),
                        inner: TypeInner::Scalar(Scalar {
                            kind: ScalarKind::Float,
                            width: 4,
                        }),
                    }),
                    Literal::F64(_) => Ok(Type {
                        name: Default::default(),
                        inner: TypeInner::Scalar(Scalar {
                            kind: ScalarKind::Float,
                            width: 8,
                        }),
                    }),
                    Literal::AbstractInt(_) | Literal::I32(_) => Ok(Type {
                        name: Default::default(),
                        inner: TypeInner::Scalar(Scalar {
                            kind: ScalarKind::Sint,
                            width: 4,
                        }),
                    }),
                    Literal::I64(_) => Ok(Type {
                        name: Default::default(),
                        inner: TypeInner::Scalar(Scalar {
                            kind: ScalarKind::Sint,
                            width: 8,
                        }),
                    }),
                    Literal::U32(_) => Ok(Type {
                        name: Default::default(),
                        inner: TypeInner::Scalar(Scalar {
                            kind: ScalarKind::Uint,
                            width: 4,
                        }),
                    }),
                    Literal::Bool(_) => Ok(Type {
                        name: Default::default(),
                        inner: TypeInner::Scalar(Scalar {
                            kind: ScalarKind::Bool,
                            width: 1,
                        }),
                    }),
                }
            }
            Expression::Math {
                fun,
                arg,
                arg1: _,
                arg2: _,
                arg3: _,
            } => {
                use MathFunction::*;
                match fun {
                    Sin => self.get_expression_type(
                        types,
                        local_variables,
                        expressions,
                        &expressions[*arg],
                    ),
                    _ => unimplemented!(),
                }
            }
            Expression::Splat { value, .. } => {
                let ty = self.get_expression_type(
                    types,
                    local_variables,
                    expressions,
                    &expressions[*value],
                )?;
                Ok(Type {
                    name: ty.name,
                    inner: ty.inner,
                })
            }
            x => Err(WriterError::UnknownExpressionType(x.clone())),
        }
    }

    fn convert_function(
        &mut self,
        module: &Module,
        handle: &Handle<Function>,
        function: &Function,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> Result<ItemFn, WriterError> {
        log::trace!("Converting function: {function:#?}");

        let is_entry_point = matches!(func_ctx.ty, back::FunctionType::EntryPoint(_));
        log::trace!("Function is entry point: {:?}", is_entry_point);

        let func_name = match func_ctx.ty {
            back::FunctionType::EntryPoint(index) => &self.names[&NameKey::EntryPoint(index)],
            back::FunctionType::Function(handle) => &self.names[&NameKey::Function(handle)],
        };
        log::trace!("Function name after namer: {:?}", func_name);

        let ident = Ident::new(func_name, Span::call_site());

        // Function arguments
        let mut inputs = function
            .arguments
            .iter()
            .enumerate()
            .map(|(index, arg)| {
                // Write argument attributes if a binding is present.
                let attrs = match arg.binding {
                    Some(Binding::BuiltIn(b)) => {
                        let attr_name = map_builtin_to_rust_gpu(&b);
                        let tokens = quote!(#attr_name);

                        vec![Attribute {
                            pound_token: token::Pound::default(),
                            style: syn::AttrStyle::Outer,
                            bracket_token: token::Bracket::default(),
                            meta: Meta::List(MetaList {
                                path: Path::from(Ident::new("spirv", Span::call_site())),
                                delimiter: syn::MacroDelimiter::Paren(Default::default()),
                                tokens,
                            }),
                        }]
                    }
                    Some(Binding::Location {
                        location,
                        interpolation,
                        sampling,
                        ..
                    }) => {
                        let location_tokens = match location {
                            // Zero is the default so we don't need to output it.
                            0 => None,
                            _ => {
                                // Convert location to an integer literal, otherwise the
                                // output is in the form of `location = 1u32` instead of
                                // `location = 1`.
                                let lit =
                                    syn::LitInt::new(&location.to_string(), Span::call_site());
                                Some(quote!(location = #lit))
                            }
                        };

                        let interpolation_tokens = match interpolation {
                            // This is the default so we don't output anything.
                            Some(Interpolation::Perspective) => None,
                            Some(Interpolation::Flat) => Some(quote!(flat)),
                            // TODO: check if this should be "noperspective" or "linear"
                            Some(Interpolation::Linear) => Some(quote!(linear)),
                            None => None,
                        };

                        let sampling_tokens = match sampling {
                            Some(Sampling::Center) => Some(quote!(center)),
                            Some(Sampling::Centroid) => Some(quote!(centroid)),
                            Some(Sampling::Sample) => Some(quote!(sample)),
                            None => None,
                        };

                        // Put all the attribute tokens together.
                        let mut total_tokens = TokenStream::new();
                        let streams: Vec<TokenStream> =
                            vec![location_tokens, interpolation_tokens, sampling_tokens]
                                .into_iter()
                                .filter_map(|x| x)
                                .collect();
                        let mut iter = streams.into_iter().peekable();
                        while let Some(stream) = iter.next() {
                            total_tokens.extend(stream);
                            if iter.peek().is_some() {
                                total_tokens.extend(quote!(, ));
                            }
                        }

                        if total_tokens.is_empty() {
                            vec![]
                        } else {
                            vec![Attribute {
                                pound_token: token::Pound::default(),
                                style: syn::AttrStyle::Outer,
                                bracket_token: token::Bracket::default(),
                                meta: Meta::List(MetaList {
                                    path: Path::from(Ident::new("spirv", Span::call_site())),
                                    delimiter: syn::MacroDelimiter::Paren(Default::default()),
                                    tokens: total_tokens,
                                }),
                            }]
                        }
                    }
                    None => vec![],
                };

                let arg_name = self.names[&func_ctx.argument_key(index as u32)].clone();
                let ty = self.convert_type(&module.types, &arg.ty)?;

                Ok(FnArg::Typed(PatType {
                    attrs,
                    pat: Box::new(Pat::Ident(PatIdent {
                        attrs: vec![],
                        by_ref: None,
                        mutability: None,
                        ident: Ident::new(&arg_name, Span::call_site()),
                        subpat: None,
                    })),
                    colon_token: token::Colon::default(),
                    ty: Box::new(ty),
                }))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Determine the return type of the function
        let return_type = match (is_entry_point, &function.result) {
            (false, Some(result)) => {
                let ty = self.convert_type(&module.types, &result.ty)?;
                ReturnType::Type(Default::default(), Box::new(ty))
            }
            (true, Some(result)) => {
                // TODO: handle binding?

                let ty = &module.types[result.ty];

                match &ty.inner {
                    TypeInner::Struct { members, .. } if ty.name.is_none() => {
                        // If the return type is an unnamed struct, treat each member as
                        // an output (which confusingly is written as an input to the
                        // entrypoint function).
                        for (i, member) in members.iter().enumerate() {
                            let member_type = self.convert_type(&module.types, &member.ty)?;

                            //let member_name = member .name .as_ref()
                            //    .ok_or(WriterError::MissingStructMemberName)?;
                            let member_name =
                                self.names[&NameKey::StructMember(result.ty, i as u32)].clone();

                            let arg_ident = Ident::new(&member_name, Span::call_site());

                            // Add a mutable reference for each struct member as an
                            // input to the entrypoint function.
                            let ref_ty = syn::Type::Reference(syn::TypeReference {
                                and_token: Default::default(),
                                lifetime: None,
                                mutability: Some(token::Mut::default()),
                                elem: Box::new(member_type),
                            });

                            inputs.push(FnArg::Typed(PatType {
                                attrs: vec![],
                                pat: Box::new(Pat::Ident(PatIdent {
                                    attrs: vec![],
                                    by_ref: None,
                                    mutability: None,
                                    ident: arg_ident,
                                    subpat: None,
                                })),
                                colon_token: Default::default(),
                                ty: Box::new(ref_ty),
                            }));
                        }
                    }
                    _ => unimplemented!(),
                }

                // Clear the return type as it's now an argument
                ReturnType::Default
            }
            (_, None) => ReturnType::Default,
        };

        let signature = Signature {
            ident,
            inputs: Punctuated::from_iter(inputs.into_iter()),
            output: return_type,
            constness: Default::default(),
            asyncness: Default::default(),
            unsafety: Default::default(),
            abi: Default::default(),
            fn_token: Default::default(),
            generics: Default::default(),
            variadic: Default::default(),
            paren_token: Default::default(),
        };

        let mut stmts = vec![];

        for (localvar_handle, _) in function.local_variables.iter() {
            let converted = self.convert_local_variable(
                module,
                handle,
                &module.types,
                &function.local_variables,
                &function.expressions,
                &localvar_handle,
            )?;
            stmts.push(converted);
        }

        for statement in &function.body {
            let mut converted = self.convert_statement(
                module,
                Some(handle),
                &module.types,
                &function.local_variables,
                &function.expressions,
                statement,
            )?;
            stmts.append(&mut converted);
        }

        let block = SynBlock {
            brace_token: Default::default(),
            stmts,
        };

        Ok(ItemFn {
            attrs: vec![],
            vis: Visibility::Inherited,
            sig: signature,
            block: Box::new(block),
        })
    }

    fn convert_local_variable(
        &mut self,
        module: &Module,
        func_handle: &Handle<Function>,
        types: &UniqueArena<Type>,
        local_variables: &Arena<LocalVariable>,
        expressions: &Arena<Expression>,
        handle: &Handle<LocalVariable>,
    ) -> Result<Stmt, WriterError> {
        log::trace!("Converting local variable");
        // TODO: handle mut vs non-mut.
        let local_variable = &local_variables[*handle];
        let ty = self.convert_type(&types, &local_variable.ty)?;
        let name = self.names[&NameKey::FunctionLocal(*func_handle, *handle)].clone();

        let init = if let Some(handle) = local_variable.init {
            log::trace!("has init");
            let e = &expressions[handle];
            log::info!("bake count: {}", e.bake_ref_count());
            // `let x = 42;`
            let expr = self.convert_nonconst_expression(module, None, expressions, &handle)?;
            Some(expr)
        } else {
            log::trace!("no init");
            // `let x;`
            None
        };

        Ok(Stmt::Local(Local {
            attrs: vec![],
            let_token: token::Let::default(),
            pat: Pat::Type(PatType {
                attrs: vec![],
                pat: Box::new(Pat::Ident(PatIdent {
                    attrs: vec![],
                    by_ref: None,
                    mutability: Some(token::Mut::default()),
                    ident: Ident::new(&name, Span::call_site()),
                    subpat: None,
                })),
                colon_token: token::Colon::default(),
                ty: Box::new(ty),
            }),
            init: init.map(|e| LocalInit {
                eq_token: token::Eq::default(),
                expr: Box::new(e),
                diverge: None,
            }),
            semi_token: token::Semi::default(),
        }))
    }

    fn convert_compound_op(
        &mut self,
        module: &Module,
        function: Option<&Handle<Function>>,
        _types: &UniqueArena<Type>,
        _local_variables: &Arena<LocalVariable>,
        expressions: &Arena<Expression>,
        op: &BinaryOperator,
        pointer: &Handle<Expression>,
        value: &Handle<Expression>,
    ) -> Result<Vec<Stmt>, WriterError> {
        log::trace!("Converting compound op");
        let pointer = self.convert_nonconst_expression(module, function, expressions, pointer)?;
        let value = self.convert_nonconst_expression(module, function, expressions, value)?;

        let compound_op = match *op {
            BinaryOperator::Add => BinOp::AddAssign(token::PlusEq::default()),
            BinaryOperator::Subtract => BinOp::SubAssign(token::MinusEq::default()),
            BinaryOperator::Multiply => BinOp::MulAssign(token::StarEq::default()),
            BinaryOperator::Divide => BinOp::DivAssign(token::SlashEq::default()),
            BinaryOperator::Modulo => BinOp::RemAssign(token::PercentEq::default()),
            BinaryOperator::And => BinOp::BitAndAssign(token::AndEq::default()),
            BinaryOperator::LogicalOr => BinOp::BitOrAssign(token::OrEq::default()),
            BinaryOperator::ExclusiveOr => BinOp::BitXorAssign(token::CaretEq::default()),
            BinaryOperator::ShiftLeft => BinOp::ShlAssign(token::ShlEq::default()),
            BinaryOperator::ShiftRight => BinOp::ShrAssign(token::ShrEq::default()),
            // Other binary operators are not compound assignments
            _ => return Err(WriterError::UnsupportedBinaryOperator(*op)),
        };

        Ok(vec![Stmt::Expr(
            Expr::Group(ExprGroup {
                attrs: vec![],
                group_token: token::Group::default(),
                expr: Box::new(Expr::Binary(syn::ExprBinary {
                    attrs: vec![],
                    left: Box::new(pointer),
                    op: compound_op,
                    right: Box::new(value),
                })),
            }),
            None,
        )])
    }

    fn convert_statement(
        &mut self,
        module: &Module,
        function: Option<&Handle<Function>>,
        types: &UniqueArena<Type>,
        local_variables: &Arena<LocalVariable>,
        expressions: &Arena<Expression>,
        statement: &Statement,
    ) -> Result<Vec<Stmt>, WriterError> {
        log::trace!("Converting statement: {statement:?}");
        match statement {
            Statement::Store { pointer, value } => {
                log::trace!("Store statement");
                let pointer_expr = &expressions[*pointer];
                let value_expr = &expressions[*value];
                log::trace!("{pointer_expr:?}");
                log::trace!("{value_expr:?}");

                if let Expression::GlobalVariable(_) = pointer_expr {
                    log::trace!("Global variable as pointer");

                    // TODO: should this check binding or address space?

                    // This is an output variable, apply dereference.
                    let left_expr =
                        self.convert_nonconst_expression(module, function, expressions, pointer)?;
                    let left_expr_deref = Expr::Unary(ExprUnary {
                        attrs: vec![],
                        op: syn::UnOp::Deref(token::Star::default()),
                        expr: Box::new(left_expr),
                    });
                    let right_expr =
                        self.convert_nonconst_expression(module, function, expressions, value)?;

                    return Ok(vec![
                        (Stmt::Expr(
                            Expr::Assign(ExprAssign {
                                attrs: vec![],
                                left: Box::new(left_expr_deref),
                                eq_token: Default::default(),
                                right: Box::new(right_expr),
                            }),
                            Some(token::Semi::default()),
                        )),
                    ]);
                }

                if let Expression::Binary { op, left, right } = value_expr {
                    log::trace!("Binary expression as value");
                    let left_expr = &expressions[*left];
                    // TODO: This incorrectly converts `x = x + 1` to `x += 1`
                    let is_compound = match left_expr {
                        Expression::Load { pointer } => {
                            let x = &expressions[*pointer];
                            x == pointer_expr
                        }
                        _ => false,
                    };
                    if is_compound {
                        log::trace!("Compound assignment");
                        return self.convert_compound_op(
                            module,
                            function,
                            types,
                            local_variables,
                            expressions,
                            op,
                            pointer,
                            right,
                        );
                    }
                }

                let left =
                    self.convert_nonconst_expression(module, function, expressions, pointer)?;
                let right =
                    self.convert_nonconst_expression(module, function, expressions, value)?;

                Ok(vec![
                    (Stmt::Expr(
                        Expr::Assign(ExprAssign {
                            attrs: vec![],
                            left: Box::new(left),
                            eq_token: Default::default(),
                            right: Box::new(right),
                        }),
                        Some(token::Semi::default()),
                    )),
                ])
            }
            Statement::If {
                condition,
                accept,
                reject,
            } => {
                log::trace!("If statement");

                // Convert the condition expression.
                let condition_expr =
                    self.convert_nonconst_expression(module, function, expressions, condition)?;

                // Convert the accept block.
                let accept_stmts: Vec<Stmt> = accept
                    .iter()
                    .map(|stmt| {
                        self.convert_statement(
                            module,
                            function,
                            types,
                            local_variables,
                            expressions,
                            stmt,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect();
                let accept_block = SynBlock {
                    brace_token: Default::default(),
                    stmts: accept_stmts,
                };

                // Convert the reject block, if it exists.
                let reject_block = if !reject.is_empty() {
                    let reject_stmts: Vec<Stmt> = reject
                        .iter()
                        .map(|stmt| {
                            self.convert_statement(
                                module,
                                function,
                                types,
                                local_variables,
                                expressions,
                                stmt,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .flatten()
                        .collect();

                    if !reject_stmts.is_empty() {
                        Some(SynBlock {
                            brace_token: Default::default(),
                            stmts: reject_stmts,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                };

                let if_stmt = Stmt::Expr(
                    Expr::If(ExprIf {
                        attrs: vec![],
                        if_token: Default::default(),
                        cond: Box::new(condition_expr),
                        then_branch: accept_block,
                        else_branch: reject_block.map(|b| {
                            (
                                token::Else::default(),
                                Box::new(Expr::Block(syn::ExprBlock {
                                    attrs: vec![],
                                    label: None,
                                    block: b,
                                })),
                            )
                        }),
                    }),
                    None,
                );

                Ok(vec![if_stmt])
            }
            Statement::Break => {
                log::trace!("Break statement");
                Ok(vec![Stmt::Expr(
                    Expr::Break(syn::ExprBreak {
                        attrs: vec![],
                        break_token: Default::default(),
                        label: None,
                        expr: None,
                    }),
                    Some(token::Semi::default()),
                )])
            }
            Statement::Loop {
                body,
                continuing,
                break_if,
            } => {
                log::trace!("Loop statement");
                // Convert loop body
                let mut body_stmts = vec![];
                for stmt in body {
                    body_stmts.append(&mut self.convert_statement(
                        module,
                        function,
                        types,
                        local_variables,
                        expressions,
                        stmt,
                    )?);
                }

                // Convert the continuing block
                let mut continuing_stmts = vec![];
                for stmt in continuing {
                    continuing_stmts.append(&mut self.convert_statement(
                        module,
                        function,
                        types,
                        local_variables,
                        expressions,
                        stmt,
                    )?);
                }

                let loop_stmt = if let Some(break_condition) = break_if {
                    log::trace!("Break condition exists");
                    let condition_expr = self.convert_nonconst_expression(
                        module,
                        function,
                        expressions,
                        break_condition,
                    )?;
                    self.convert_conditional_break_loop(condition_expr, body_stmts)?
                } else {
                    log::trace!("No break condition");
                    // Check if the first statement is an `If` containing only a
                    // `Break`. If so, we can convert the loop to a while loop.
                    if let Some(first) = body_stmts.first() {
                        match first {
                            Stmt::Expr(
                                Expr::If(ExprIf {
                                    cond,
                                    then_branch: SynBlock { stmts, .. },
                                    else_branch: None,
                                    ..
                                }),
                                _,
                            ) => {
                                if stmts.len() == 1 {
                                    if let Some(first) = stmts.first() {
                                        if matches!(first, Stmt::Expr(Expr::Break(_), _)) {
                                            log::trace!("while loop detected");
                                            // If we get here we know the first body statement is:
                                            // * An `if`
                                            // * ... that does not have an `else`
                                            // * ... has one statement in its body
                                            // * ... and that statement is a break

                                            // We need to invert the condition.
                                            let cond = match cond.as_ref() {
                                                Expr::Unary(ExprUnary {
                                                    op: UnOp::Not(..),
                                                    expr,
                                                    ..
                                                }) => {
                                                    // We need to remove the top level
                                                    // parens if they exist as they are
                                                    // unnecessary after removing the
                                                    // `Not` operation.
                                                    //
                                                    // This isn't strictly necessary but
                                                    // will prevent warnings from
                                                    // showing up when the output is
                                                    // compiled.
                                                    match expr.as_ref() {
                                                        Expr::Paren(ExprParen { expr, .. }) => expr,
                                                        x => x,
                                                    }
                                                }
                                                x => x,
                                            };

                                            return Ok(vec![Stmt::Expr(
                                                Expr::While(syn::ExprWhile {
                                                    attrs: vec![],
                                                    label: None,
                                                    while_token: Default::default(),
                                                    cond: Box::new(cond.clone()),
                                                    body: SynBlock {
                                                        brace_token: Default::default(),
                                                        stmts: body_stmts[1..].to_vec(),
                                                    },
                                                }),
                                                None,
                                            )]);
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    self.convert_infinite_loop(body_stmts)?
                };

                Ok(vec![loop_stmt])
            }
            Statement::Emit(ref _range) => {
                log::trace!("Emit statement");
                Ok(vec![])
            }
            Statement::Return { value } => {
                log::trace!("Return statement");
                if let Some(value_handle) = value {
                    let value_expr = self.convert_nonconst_expression(
                        module,
                        function,
                        expressions,
                        value_handle,
                    )?;
                    Ok(vec![Stmt::Expr(
                        Expr::Return(ExprReturn {
                            attrs: vec![],
                            return_token: token::Return::default(),
                            expr: Some(Box::new(value_expr)),
                        }),
                        Some(token::Semi::default()),
                    )])
                } else {
                    Ok(vec![])
                }
            }
            Statement::Block(block_statements) => {
                log::trace!("Block statement");
                // TODO: should this be a syn block?

                // Create a vector to hold the converted Rust statements
                let mut stmts = vec![];

                // Iterate over each statement in the block and convert it
                for stmt in block_statements {
                    stmts.append(&mut self.convert_statement(
                        module,
                        function,
                        types,
                        local_variables,
                        expressions,
                        stmt,
                    )?);
                }

                Ok(stmts)
            }
            Statement::Continue => Ok(vec![Stmt::Expr(
                Expr::Continue(syn::ExprContinue {
                    attrs: vec![],
                    continue_token: token::Continue::default(),
                    label: None,
                }),
                Some(token::Semi::default()),
            )]),
            Statement::Call {
                function,
                arguments,
                result,
            } => {
                let func_name = &self.names[&NameKey::Function(*function)];
                let func_ident = Ident::new(func_name, Span::call_site());
                log::trace!("Call statement: {func_name}");

                // Convert the arguments
                let args_exprs = arguments
                    .iter()
                    .map(|arg| {
                        self.convert_nonconst_expression(module, Some(function), expressions, arg)
                    })
                    .collect::<Result<Vec<_>, _>>();

                let args_exprs = args_exprs?;

                // Construct the function call expression
                let func_expr = Expr::Path(ExprPath {
                    attrs: vec![],
                    qself: None,
                    path: Path::from(func_ident),
                });

                let punctuated_args = Punctuated::from_iter(args_exprs.into_iter());

                let call_expr = Expr::Call(ExprCall {
                    attrs: vec![],
                    func: Box::new(func_expr),
                    paren_token: token::Paren::default(),
                    args: punctuated_args,
                });

                // Handle the result assignment if applicable
                let stmt_expr = if let Some(result_handle) = result {
                    let _result_expr = self.convert_nonconst_expression(
                        module,
                        Some(function),
                        expressions,
                        result_handle,
                    )?;
                    log::debug!("{:?}", expressions[*result_handle]);
                    Expr::Block(syn::ExprBlock {
                        attrs: vec![],
                        label: None,
                        block: SynBlock {
                            brace_token: token::Brace::default(),
                            stmts: vec![],
                        },
                    })
                } else {
                    call_expr
                };

                Ok(vec![Stmt::Expr(stmt_expr, Some(token::Semi::default()))])
            }
            x => todo!("{x:?}"),
        }
    }

    fn convert_possibly_const_expression<E>(
        &mut self,
        module: &Module,
        function: Option<&Handle<Function>>,
        expr: &Handle<Expression>,
        local_variables: &Arena<LocalVariable>,
        expressions: &Arena<Expression>,
        convert_expression: E,
    ) -> Result<Expr, WriterError>
    where
        E: Fn(&mut Self, &Handle<Expression>) -> Result<Expr, WriterError>,
    {
        log::trace!("Converting expression: {:#?}", &expressions[*expr]);
        match &expressions[*expr] {
            Expression::Constant(handle) => {
                // Handle constant expressions. We need to retrieve the constant value
                // and type from 'handle' and convert it to a Rust literal.
                log::trace!("Constant variable expression");
                let constant = &module.constants[*handle];
                if constant.name.is_some() {
                    Ok(syn::Expr::Path(ExprPath {
                        attrs: vec![],
                        qself: None,
                        path: Path::from(Ident::new(
                            &self.names[&NameKey::Constant(*handle)],
                            Span::call_site(),
                        )),
                    }))
                } else {
                    self.convert_const_expression(module, &constant.init)
                }
            }
            Expression::FunctionArgument(pos) => {
                let name = self.names
                    [&NameKey::FunctionArgument(*function.expect("arg has function"), *pos)]
                    .clone();
                Ok(Expr::Path(ExprPath {
                    attrs: vec![],
                    qself: None,
                    path: Path::from(Ident::new(&name, Span::call_site())),
                }))
            }
            Expression::CallResult(handle) => {
                let func = &module.functions[*handle];
                log::trace!("Converting call result for function {:?}", func.name);
                log::trace!("Function data: {:?}", func);

                Ok(Expr::Block(syn::ExprBlock {
                    attrs: vec![],
                    label: None,
                    block: SynBlock {
                        brace_token: token::Brace::default(),
                        stmts: vec![],
                    },
                }))
            }
            Expression::GlobalVariable(handle) => {
                let global_var = &module.global_variables[*handle];
                log::trace!("Global variable expression: {:?}", global_var);

                let name: String = self.names[&NameKey::GlobalVariable(*handle)].clone();

                Ok(syn::Expr::Path(ExprPath {
                    attrs: vec![],
                    qself: None,
                    path: Path::from(Ident::new(&name, Span::call_site())),
                }))
            }
            Expression::Access { base, index } => {
                let base_expr =
                    self.convert_nonconst_expression(module, function, expressions, base)?;

                let index_expr =
                    self.convert_nonconst_expression(module, function, expressions, index)?;

                let base_type = self.get_expression_type(
                    &module.types,
                    local_variables,
                    expressions,
                    &expressions[*base],
                )?;

                match &base_type.inner {
                    TypeInner::Vector { size, .. } => {
                        // Handling vector access
                        if let Expr::Lit(ExprLit {
                            lit: syn::Lit::Int(lit_int),
                            ..
                        }) = index_expr
                        {
                            let index_value = lit_int.base10_parse::<u32>().unwrap();

                            let component = match index_value {
                                0 if *size >= VectorSize::Bi => "x",
                                1 if *size >= VectorSize::Bi => "y",
                                2 if *size >= VectorSize::Tri => "z",
                                3 if *size >= VectorSize::Quad => "w",
                                _ => {
                                    return Err(WriterError::InvalidVectorIndex(*size, index_value))
                                }
                            };

                            Ok(Expr::Field(ExprField {
                                attrs: vec![],
                                base: Box::new(base_expr),
                                dot_token: Default::default(),
                                member: Member::Named(Ident::new(component, Span::call_site())),
                            }))
                        } else {
                            todo!("Non-literal index in vector access")
                        }
                    }
                    TypeInner::Matrix { columns, rows, .. } => {
                        // Handling matrix access
                        if let Expr::Lit(ExprLit {
                            lit: syn::Lit::Int(lit_int),
                            ..
                        }) = index_expr
                        {
                            let index_value = lit_int.base10_parse::<u32>().unwrap();

                            let column_access_field = match index_value {
                                0 if *columns as u32 >= 1 => "x_axis",
                                1 if *columns as u32 >= 2 => "y_axis",
                                2 if *columns as u32 >= 3 => "z_axis",
                                3 if *columns as u32 >= 4 => "w_axis",
                                _ => {
                                    return Err(WriterError::InvalidMatrixIndex(
                                        *columns,
                                        *rows,
                                        index_value,
                                    ))
                                }
                            };

                            Ok(Expr::Field(ExprField {
                                attrs: vec![],
                                base: Box::new(base_expr),
                                dot_token: Default::default(),
                                member: Member::Named(Ident::new(
                                    column_access_field,
                                    Span::call_site(),
                                )),
                            }))
                        } else {
                            todo!("Non-literal index in matrix access")
                        }
                    }
                    // Add handling for other types if necessary
                    _ => todo!("Handle Access for type {base_type:?}"),
                }
            }
            Expression::AccessIndex { base, index } => {
                log::trace!("Access index");
                let base_expr = self.convert_possibly_const_expression(
                    module,
                    function,
                    &base,
                    local_variables,
                    expressions,
                    convert_expression,
                )?;

                let base_type = self.get_expression_type(
                    &module.types,
                    local_variables,
                    expressions,
                    &expressions[*base],
                )?;

                match &base_type.inner {
                    TypeInner::Vector { size, .. } => {
                        let swizzle = match index {
                            0 if *size >= VectorSize::Bi => "x",
                            1 if *size >= VectorSize::Bi => "y",
                            2 if *size >= VectorSize::Tri => "z",
                            3 if *size >= VectorSize::Quad => "w",
                            _ => return Err(WriterError::InvalidVectorIndex(*size, *index)),
                        };

                        log::trace!("swizzle index: {swizzle}");

                        Ok(Expr::Field(ExprField {
                            attrs: vec![],
                            base: Box::new(base_expr),
                            dot_token: Default::default(),
                            member: Member::Named(Ident::new(swizzle, Span::call_site())),
                        }))
                    }
                    TypeInner::Matrix { columns, rows, .. } => {
                        let column_access_field = match index {
                            0 if *columns as u32 >= 1 => "x_axis",
                            1 if *columns as u32 >= 2 => "y_axis",
                            2 if *columns as u32 >= 3 => "z_axis",
                            3 if *columns as u32 >= 4 => "w_axis",
                            _ => {
                                return Err(WriterError::InvalidMatrixIndex(
                                    *columns, *rows, *index,
                                ))
                            }
                        };

                        Ok(Expr::Field(ExprField {
                            attrs: vec![],
                            base: Box::new(base_expr),
                            dot_token: Default::default(),
                            member: Member::Named(Ident::new(
                                column_access_field,
                                Span::call_site(),
                            )),
                        }))
                    }
                    _ => todo!("accessindex"),
                }
            }
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                log::trace!("Swizzle");
                let vector_expr = self.convert_possibly_const_expression(
                    module,
                    function,
                    &vector,
                    local_variables,
                    expressions,
                    convert_expression,
                )?;
                let isize = match size {
                    VectorSize::Bi => 2,
                    VectorSize::Tri => 3,
                    VectorSize::Quad => 4,
                };
                let method_name: String = pattern[..isize]
                    .iter()
                    .map(|x| match x {
                        SwizzleComponent::X => 'x',
                        SwizzleComponent::Y => 'y',
                        SwizzleComponent::Z => 'z',
                        SwizzleComponent::W => 'w',
                    })
                    .collect();

                log::trace!("Swizzle method: {method_name}");

                Ok(Expr::MethodCall(ExprMethodCall {
                    attrs: vec![],
                    receiver: Box::new(vector_expr),
                    method: Ident::new(&method_name, Span::call_site()),
                    turbofish: None,
                    dot_token: token::Dot::default(),
                    paren_token: token::Paren::default(),
                    args: Punctuated::new(),
                }))
            }
            Expression::Splat { value, size } => {
                let value_expr = convert_expression(self, &value)?;

                let ty = self.get_expression_type(
                    &module.types,
                    local_variables,
                    expressions,
                    &expressions[*value],
                )?;
                let ty = map_type_to_glam(&ty)?;

                let ty = match (size, ty.as_str()) {
                    (VectorSize::Bi, "f32") => "Vec2",
                    (VectorSize::Tri, "f32") => "Vec3",
                    (VectorSize::Quad, "f32") => "Vec4",
                    (VectorSize::Bi, "f64") => "DVec2",
                    (VectorSize::Tri, "f64") => "DVec3",
                    (VectorSize::Quad, "f64") => "DVec4",
                    (VectorSize::Bi, "i16") => "I16Vec2",
                    (VectorSize::Tri, "i16") => "I16Vec3",
                    (VectorSize::Quad, "i16") => "I16Vec4",
                    (VectorSize::Bi, "u16") => "U16Vec2",
                    (VectorSize::Tri, "u16") => "U16Vec3",
                    (VectorSize::Quad, "u16") => "U16Vec4",
                    (VectorSize::Bi, "i32") => "IVec2",
                    (VectorSize::Tri, "i32") => "IVec3",
                    (VectorSize::Quad, "i32") => "IVec4",
                    (VectorSize::Bi, "u32") => "UVec2",
                    (VectorSize::Tri, "u32") => "UVec3",
                    (VectorSize::Quad, "u32") => "UVec4",
                    (VectorSize::Bi, "i64") => "I64Vec2",
                    (VectorSize::Tri, "i64") => "I64Vec3",
                    (VectorSize::Quad, "i64") => "I64Vec4",
                    (VectorSize::Bi, "u64") => "U64Vec2",
                    (VectorSize::Tri, "u64") => "U64Vec3",
                    (VectorSize::Quad, "u64") => "U64Vec4",
                    (VectorSize::Bi, "bool") => "BVec2",
                    (VectorSize::Tri, "bool") => "BVec3",
                    (VectorSize::Quad, "bool") => "BVec4",
                    _ => unimplemented!(),
                };

                let path_segments = vec![
                    PathSegment::from(Ident::new("glam", Span::call_site())),
                    PathSegment::from(Ident::new(&ty, Span::call_site())),
                    PathSegment::from(Ident::new("splat", Span::call_site())),
                ];

                let path = Path {
                    leading_colon: None,
                    segments: Punctuated::from_iter(path_segments),
                };

                let func = ExprPath {
                    attrs: vec![],
                    qself: None,
                    path,
                };

                let call = ExprCall {
                    attrs: vec![],
                    func: Box::new(Expr::Path(func)),
                    paren_token: token::Paren::default(),
                    args: Punctuated::from_iter(vec![value_expr]),
                };

                Ok(Expr::Call(call))
            }
            Expression::Compose { ty, components } => {
                let ty = &module.types[*ty];

                let component_exprs: Result<Vec<_>, _> = components
                    .iter()
                    .map(|comp| convert_expression(self, comp))
                    .collect();

                let component_exprs = component_exprs?;

                match &ty.inner {
                    &TypeInner::Vector { .. } => {
                        let ty_str = map_type_to_glam(ty)?;
                        // Handle vector composition
                        let path_segments = vec![
                            Ident::new("glam", Span::call_site()),
                            Ident::new(&ty_str, Span::call_site()),
                            Ident::new("new", Span::call_site()),
                        ]
                        .into_iter()
                        .map(PathSegment::from)
                        .collect();

                        let path = Path {
                            leading_colon: None,
                            segments: path_segments,
                        };

                        let func = ExprPath {
                            attrs: vec![],
                            qself: None,
                            path,
                        };
                        let args = component_exprs.into_iter().collect::<Vec<_>>();

                        let punctuated_args =
                            Punctuated::<Expr, token::Comma>::from_iter(args.iter().cloned());

                        let call = ExprCall {
                            attrs: vec![],
                            func: Box::new(Expr::Path(func)),
                            paren_token: token::Paren::default(),
                            args: punctuated_args,
                        };

                        Ok(syn::Expr::Call(call))
                    }
                    &TypeInner::Matrix { .. } => {
                        let ty_str = map_type_to_glam(ty)?;
                        // Handle matrix composition
                        let path_segments = vec![
                            Ident::new("glam", Span::call_site()),
                            Ident::new(&ty_str, Span::call_site()),
                            Ident::new("from_cols", Span::call_site()),
                        ]
                        .into_iter()
                        .map(PathSegment::from)
                        .collect();

                        let path = Path {
                            leading_colon: None,
                            segments: path_segments,
                        };

                        let func = ExprPath {
                            attrs: vec![],
                            qself: None,
                            path,
                        };

                        let punctuated_args = Punctuated::<Expr, token::Comma>::from_iter(
                            component_exprs.iter().cloned(),
                        );

                        let call = ExprCall {
                            attrs: vec![],
                            func: Box::new(Expr::Path(func)),
                            paren_token: token::Paren::default(),
                            args: punctuated_args,
                        };

                        Ok(syn::Expr::Call(call))
                    }
                    &TypeInner::Struct { ref members, .. } => {
                        let struct_name = if let Some(n) = &ty.name {
                            n.clone()
                        } else {
                            assert_eq!(members.len(), 1);
                            members
                                .first()
                                .expect("exactly one struct member")
                                .clone()
                                .name
                                .ok_or(WriterError::MissingStructMemberName)?
                        };

                        let ident = syn::Ident::new(&struct_name, Span::call_site());

                        // Create a path for the struct type
                        let path = Path {
                            leading_colon: None,
                            segments: Punctuated::from_iter(vec![PathSegment::from(ident)]),
                        };
                        Ok(syn::Expr::Path(ExprPath {
                            attrs: vec![],
                            qself: None,
                            path,
                        }))
                    }
                    // Handle other types if necessary
                    x => todo!("Handle type in composition: {x:?}"),
                }
            }
            Expression::Literal(literal) => match literal {
                Literal::F64(value) => Ok(Expr::Lit(ExprLit {
                    attrs: vec![],
                    lit: syn::Lit::Float(syn::LitFloat::new(
                        &format!("{:.1}", value),
                        Span::call_site(),
                    )),
                })),
                Literal::F32(value) => Ok(Expr::Lit(ExprLit {
                    attrs: vec![],
                    lit: syn::Lit::Float(syn::LitFloat::new(
                        &format!("{:.1}", value),
                        Span::call_site(),
                    )),
                })),
                Literal::U32(value) => Ok(Expr::Lit(ExprLit {
                    attrs: vec![],
                    lit: syn::Lit::Int(syn::LitInt::new(&value.to_string(), Span::call_site())),
                })),
                Literal::I32(value) => Ok(Expr::Lit(ExprLit {
                    attrs: vec![],
                    lit: syn::Lit::Int(syn::LitInt::new(&value.to_string(), Span::call_site())),
                })),
                Literal::I64(value) => Ok(Expr::Lit(ExprLit {
                    attrs: vec![],
                    lit: syn::Lit::Int(syn::LitInt::new(&value.to_string(), Span::call_site())),
                })),
                Literal::Bool(value) => Ok(Expr::Lit(ExprLit {
                    attrs: vec![],
                    lit: syn::Lit::Bool(syn::LitBool::new(*value, Span::call_site())),
                })),
                x => todo!("{x:?}"),
            },
            Expression::Binary { op, left, right } => {
                log::trace!("binary");
                let left_expr =
                    self.convert_nonconst_expression(module, function, expressions, left)?;
                let right_expr = self.convert_possibly_const_expression(
                    module,
                    function,
                    &right,
                    local_variables,
                    expressions,
                    convert_expression,
                )?;

                let binary_op = match op {
                    BinaryOperator::Add => BinOp::Add(token::Plus::default()),
                    BinaryOperator::Subtract => BinOp::Sub(token::Minus::default()),
                    BinaryOperator::Multiply => BinOp::Mul(token::Star::default()),
                    BinaryOperator::Divide => BinOp::Div(token::Slash::default()),
                    BinaryOperator::Modulo => BinOp::Rem(token::Percent::default()),
                    BinaryOperator::Equal => BinOp::Eq(token::EqEq::default()),
                    BinaryOperator::NotEqual => BinOp::Ne(token::Ne::default()),
                    BinaryOperator::Less => BinOp::Lt(token::Lt::default()),
                    BinaryOperator::LessEqual => BinOp::Le(token::Le::default()),
                    BinaryOperator::Greater => BinOp::Gt(token::Gt::default()),
                    BinaryOperator::GreaterEqual => BinOp::Ge(token::Ge::default()),
                    BinaryOperator::And => BinOp::BitAnd(token::And::default()),
                    BinaryOperator::ExclusiveOr => BinOp::BitXor(token::Caret::default()),
                    BinaryOperator::InclusiveOr => BinOp::BitOr(token::Or::default()),
                    BinaryOperator::LogicalAnd => BinOp::And(token::AndAnd::default()),
                    BinaryOperator::LogicalOr => BinOp::Or(token::OrOr::default()),
                    BinaryOperator::ShiftLeft => BinOp::Shl(token::Shl::default()),
                    BinaryOperator::ShiftRight => BinOp::Shr(token::Shr::default()),
                };

                Ok(Expr::Group(ExprGroup {
                    attrs: vec![],
                    group_token: token::Group::default(),
                    expr: Box::new(Expr::Binary(syn::ExprBinary {
                        attrs: vec![],
                        left: Box::new(left_expr),
                        op: binary_op,
                        right: Box::new(right_expr),
                    })),
                }))
            }
            Expression::Unary { op, expr } => {
                log::trace!("unary");
                let operand_expr = self.convert_possibly_const_expression(
                    module,
                    function,
                    &expr,
                    local_variables,
                    expressions,
                    convert_expression,
                )?;

                let unary_op = match op {
                    UnaryOperator::Negate => syn::UnOp::Neg(token::Minus::default()),
                    UnaryOperator::LogicalNot => syn::UnOp::Not(token::Not::default()),
                    UnaryOperator::BitwiseNot => syn::UnOp::Not(token::Not::default()),
                };

                let operand_expr = match operand_expr {
                    x @ Expr::Group(_) => Expr::Paren(ExprParen {
                        attrs: vec![],
                        paren_token: token::Paren::default(),
                        expr: Box::new(x),
                    }),
                    x => x,
                };

                Ok(Expr::Unary(ExprUnary {
                    attrs: vec![],
                    op: unary_op,
                    expr: Box::new(operand_expr),
                }))
            }
            Expression::LocalVariable(handle) => {
                let local_var = &local_variables[*handle];
                log::trace!("local variable expression: {:?}", local_var.name);

                // TODO: switch on entrypoint?

                let name = self.names
                    [&NameKey::FunctionLocal(*function.expect("local var has function"), *handle)]
                    .clone();
                Ok(syn::Expr::Path(ExprPath {
                    attrs: vec![],
                    qself: None,
                    path: Path::from(Ident::new(&name, Span::call_site())),
                }))
            }
            Expression::Load { pointer } => {
                log::trace!("load expression");

                // Convert the pointer expression
                let loaded_expr = self.convert_possibly_const_expression(
                    module,
                    function,
                    &pointer,
                    local_variables,
                    expressions,
                    convert_expression,
                )?;

                Ok(loaded_expr)
            }
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                ..
            } => {
                let arg_expr =
                    self.convert_nonconst_expression(module, function, expressions, arg)?;
                let arg_expr = match arg_expr {
                    x @ Expr::Group(_) => Expr::Paren(ExprParen {
                        attrs: vec![],
                        paren_token: token::Paren::default(),
                        expr: Box::new(x),
                    }),
                    x => x,
                };

                let arg_type = self.get_expression_type(
                    &module.types,
                    local_variables,
                    expressions,
                    &expressions[*arg],
                )?;

                let (func_name, mut additional_args) = match fun {
                    MathFunction::Abs => ("abs", vec![]),
                    MathFunction::Min => (
                        "min",
                        vec![arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?],
                    ),
                    MathFunction::Max => (
                        "max",
                        vec![arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?],
                    ),
                    MathFunction::Clamp => (
                        "clamp",
                        vec![
                            arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?,
                            arg2.ok_or(WriterError::MissingMathFunctionArgument(*fun))?,
                        ],
                    ),
                    MathFunction::Cos => ("cos", vec![]),
                    MathFunction::Cosh => ("cosh", vec![]),
                    MathFunction::Sin => ("sin", vec![]),
                    MathFunction::Sinh => ("sinh", vec![]),
                    MathFunction::Tan => ("tan", vec![]),
                    MathFunction::Tanh => ("tanh", vec![]),
                    MathFunction::Acos => ("acos", vec![]),
                    MathFunction::Asin => ("asin", vec![]),
                    MathFunction::Atan => ("atan", vec![]),
                    MathFunction::Atan2 => (
                        "atan2",
                        vec![arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?],
                    ),
                    MathFunction::Pow => (
                        "powf",
                        vec![arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?],
                    ),
                    MathFunction::Exp => ("exp", vec![]),
                    MathFunction::Exp2 => ("exp2", vec![]),
                    MathFunction::Log => ("ln", vec![]), // Natural logarithm
                    MathFunction::Log2 => ("log2", vec![]),
                    MathFunction::Sqrt => ("sqrt", vec![]),
                    MathFunction::InverseSqrt => ("inversesqrt", vec![]),
                    MathFunction::Radians => ("to_radians", vec![]),
                    MathFunction::Degrees => ("to_degrees", vec![]),
                    MathFunction::Ceil => ("ceil", vec![]),
                    MathFunction::Floor => ("floor", vec![]),
                    MathFunction::Round => ("round", vec![]),
                    MathFunction::Fract => ("fract", vec![]),
                    MathFunction::Trunc => ("trunc", vec![]),
                    MathFunction::Modf => ("modf", vec![]),
                    MathFunction::Frexp => ("frexp", vec![]),
                    MathFunction::Ldexp => (
                        "ldexp",
                        vec![arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?],
                    ),
                    MathFunction::FaceForward => (
                        "face_forward",
                        vec![
                            arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?,
                            arg2.ok_or(WriterError::MissingMathFunctionArgument(*fun))?,
                        ],
                    ),
                    MathFunction::Reflect => (
                        "reflect",
                        vec![arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?],
                    ),
                    MathFunction::Refract => (
                        "refract",
                        vec![
                            arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?,
                            arg2.ok_or(WriterError::MissingMathFunctionArgument(*fun))?,
                        ],
                    ),
                    MathFunction::Sign => ("sign", vec![]),
                    MathFunction::Fma => (
                        "fma",
                        vec![
                            arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?,
                            arg2.ok_or(WriterError::MissingMathFunctionArgument(*fun))?,
                        ],
                    ),
                    MathFunction::Mix => (
                        "lerp",
                        vec![
                            arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?,
                            arg2.ok_or(WriterError::MissingMathFunctionArgument(*fun))?,
                        ],
                    ),
                    MathFunction::Step => (
                        "step",
                        vec![arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?],
                    ),
                    MathFunction::SmoothStep => (
                        "smooth_step",
                        vec![
                            arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?,
                            arg2.ok_or(WriterError::MissingMathFunctionArgument(*fun))?,
                        ],
                    ),
                    MathFunction::Distance => (
                        "distance",
                        vec![arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?],
                    ),
                    MathFunction::Length => ("length", vec![]),
                    MathFunction::Normalize => ("normalize", vec![]),
                    MathFunction::Outer => (
                        "outer",
                        vec![arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?],
                    ),
                    MathFunction::Cross => (
                        "cross",
                        vec![arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?],
                    ),
                    MathFunction::Dot => (
                        "dot",
                        vec![arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?],
                    ),
                    MathFunction::Saturate => ("saturate", vec![]),
                    MathFunction::Determinant => ("determinant", vec![]),
                    MathFunction::Transpose => ("transpose", vec![]),
                    MathFunction::Inverse => ("inverse", vec![]),
                    MathFunction::Pack4x8snorm => ("pack4x8snorm", vec![]),
                    MathFunction::Pack4x8unorm => ("pack4x8unorm", vec![]),
                    MathFunction::Pack2x16snorm => ("pack2x16snorm", vec![]),
                    MathFunction::Pack2x16unorm => ("pack2x16unorm", vec![]),
                    MathFunction::Pack2x16float => ("pack2x16float", vec![]),
                    MathFunction::Unpack4x8snorm => ("unpack4x8snorm", vec![]),
                    MathFunction::Unpack4x8unorm => ("unpack4x8unorm", vec![]),
                    MathFunction::Unpack2x16snorm => ("unpack2x16snorm", vec![]),
                    MathFunction::Unpack2x16unorm => ("unpack2x16unorm", vec![]),
                    MathFunction::Unpack2x16float => ("unpack2x16float", vec![]),
                    MathFunction::CountTrailingZeros => ("count_trailing_zeros", vec![]),
                    MathFunction::CountLeadingZeros => ("count_leading_zeros", vec![]),
                    MathFunction::CountOneBits => ("count_one_bits", vec![]),
                    MathFunction::ReverseBits => ("reverse_bits", vec![]),
                    MathFunction::ExtractBits => (
                        "extract_bits",
                        vec![
                            arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?,
                            arg2.ok_or(WriterError::MissingMathFunctionArgument(*fun))?,
                        ],
                    ),
                    MathFunction::InsertBits => (
                        "insert_bits",
                        vec![
                            arg1.ok_or(WriterError::MissingMathFunctionArgument(*fun))?,
                            arg2.ok_or(WriterError::MissingMathFunctionArgument(*fun))?,
                        ],
                    ),
                    MathFunction::FindLsb => ("find_lsb", vec![]),
                    MathFunction::FindMsb => ("find_msb", vec![]),
                    MathFunction::Asinh => ("asinh", vec![]),
                    MathFunction::Acosh => ("acosh", vec![]),
                    MathFunction::Atanh => ("atanh", vec![]),
                };

                // Check to see if we need to decompose and recompose on the Rust
                // side. Some GLSL functions take vectors where the glam or std
                // version only takes scalars.
                let operate_on_parts = match fun {
                    // These functions always operate on scalars on the rust side, so we
                    // need to operate on their parts.
                    MathFunction::Pow | MathFunction::Sin => true,
                    // These functions conditionally operate on parts.
                    MathFunction::Mix => {
                        let factor_handle = additional_args.last_mut().expect("argument handle");
                        let factor_expr = &expressions[*factor_handle];
                        let ty = self.get_expression_type(
                            &module.types,
                            local_variables,
                            expressions,
                            &expressions[*factor_handle],
                        )?;
                        match (factor_expr, ty.inner) {
                            // It's a splat, just pass in the scalar directly.
                            (Expression::Splat { value, .. }, _) => {
                                *factor_handle = *value;
                                false
                            }
                            // It's a complex type, operate on parts.
                            (_, TypeInner::Vector { .. }) | (_, TypeInner::Matrix { .. }) => true,
                            // By default, assume the function can be called with
                            // argument as-is.
                            _ => false,
                        }
                    }
                    // By default assume functions can operate on their arguments as-is.
                    _ => false,
                };

                // Convert additional argument handles to Expr
                let mut args_exprs: Vec<(Expr, Type)> = vec![];
                for handle in additional_args {
                    let additional_arg_expr =
                        self.convert_nonconst_expression(module, function, expressions, &handle)?;
                    // We need to do a group so parens are handled correctly.
                    let additional_arg_expr = Expr::Group(ExprGroup {
                        attrs: vec![],
                        group_token: token::Group::default(),
                        expr: Box::new(additional_arg_expr),
                    });
                    let additional_arg_type = self.get_expression_type(
                        &module.types,
                        local_variables,
                        expressions,
                        &expressions[handle],
                    )?;

                    args_exprs.push((additional_arg_expr, additional_arg_type));
                }

                let expr = match (arg_type.inner.clone(), operate_on_parts) {
                    // Call the method directly on scalars or complex args that the method can handle.
                    (TypeInner::Scalar { .. }, _)
                    | (TypeInner::Vector { .. }, false)
                    | (TypeInner::Matrix { .. }, false) => Expr::MethodCall(ExprMethodCall {
                        attrs: vec![],
                        receiver: Box::new(Expr::Group(ExprGroup {
                            attrs: vec![],
                            group_token: token::Group::default(),
                            expr: Box::new(arg_expr),
                        })),
                        method: Ident::new(func_name, Span::call_site()),
                        turbofish: None,
                        dot_token: token::Dot::default(),
                        paren_token: token::Paren::default(),
                        args: Punctuated::from_iter(
                            args_exprs.iter().cloned().map(|(expr, _ty)| expr),
                        ),
                    }),
                    // Call the method on parts of the the vector and then reassemble to
                    // a new vector.
                    (TypeInner::Vector { size, scalar }, true) => {
                        let component_names = match size {
                            VectorSize::Bi => vec!["x", "y"],
                            VectorSize::Tri => vec!["x", "y", "z"],
                            VectorSize::Quad => vec!["x", "y", "z", "w"],
                        };

                        // Create expressions for each component.
                        let mut component_exprs: Vec<_> = vec![];
                        for name in component_names {
                            let mut method_args: Vec<Expr> = vec![];
                            for (expr, ty) in args_exprs.iter() {
                                let method_arg = match ty.inner {
                                    TypeInner::Scalar { .. } => Ok(expr.clone()),
                                    TypeInner::Vector {
                                        size: arg_size,
                                        scalar: arg_scalar,
                                    } => {
                                        if arg_size != size {
                                            return Err(WriterError::MismatchedVectorArgSize(
                                                size, arg_size,
                                            ));
                                        }
                                        if arg_scalar != scalar {
                                            return Err(
                                                WriterError::MismatchedVectorArgScalarType(
                                                    scalar, arg_scalar,
                                                ),
                                            );
                                        }
                                        Ok(Expr::Field(ExprField {
                                            attrs: vec![],
                                            base: Box::new(expr.clone()),
                                            dot_token: Default::default(),
                                            member: Member::Named(Ident::new(
                                                name,
                                                Span::call_site(),
                                            )),
                                        }))
                                    }
                                    _ => todo!(),
                                }?;
                                method_args.push(method_arg);
                            }
                            let base = Expr::Field(ExprField {
                                attrs: vec![],
                                base: Box::new(Expr::Group(ExprGroup {
                                    attrs: vec![],
                                    group_token: token::Group::default(),
                                    expr: Box::new(arg_expr.clone()),
                                })),
                                dot_token: Default::default(),
                                member: Member::Named(Ident::new(name, Span::call_site())),
                            });

                            let func_call = Expr::MethodCall(ExprMethodCall {
                                attrs: vec![],
                                receiver: Box::new(base),
                                method: Ident::new(func_name, Span::call_site()),
                                turbofish: None,
                                dot_token: token::Dot::default(),
                                paren_token: token::Paren::default(),
                                args: Punctuated::from_iter(method_args.iter().cloned()),
                            });

                            component_exprs.push(Expr::Group(ExprGroup {
                                attrs: vec![],
                                group_token: token::Group::default(),
                                expr: Box::new(func_call),
                            }));
                        }

                        // Construct a new vector with these expressions
                        let path_segments = vec![
                            Ident::new("glam", Span::call_site()),
                            Ident::new(&map_type_to_glam(&arg_type)?, Span::call_site()),
                            Ident::new("new", Span::call_site()),
                        ]
                        .into_iter()
                        .map(PathSegment::from)
                        .collect();

                        let path = Path {
                            leading_colon: None,
                            segments: path_segments,
                        };

                        Expr::Call(ExprCall {
                            attrs: vec![],
                            func: Box::new(Expr::Path(ExprPath {
                                attrs: vec![],
                                qself: None,
                                path,
                            })),
                            paren_token: token::Paren::default(),
                            args: Punctuated::from_iter(component_exprs),
                        })
                    }
                    // Call the method on parts of the the matrix and then reassemble to
                    // a new matrix.
                    (
                        TypeInner::Matrix {
                            columns,
                            rows,
                            scalar,
                        },
                        true,
                    ) => {
                        let element_count = match (columns, rows) {
                            (VectorSize::Bi, VectorSize::Bi) => 4,
                            (VectorSize::Tri, VectorSize::Tri) => 9,
                            (VectorSize::Quad, VectorSize::Quad) => 16,
                            _ => {
                                return Err(WriterError::UnsupportedMatrixType(
                                    columns, rows, scalar,
                                ))
                            }
                        };

                        match fun {
                            MathFunction::Pow | MathFunction::Sin => {
                                let matrix_to_array_expr = Expr::MethodCall(ExprMethodCall {
                                    attrs: vec![],
                                    receiver: Box::new(arg_expr.clone()),
                                    method: Ident::new("to_cols_array", Span::call_site()),
                                    turbofish: None,
                                    dot_token: Default::default(),
                                    paren_token: token::Paren::default(),
                                    args: Punctuated::new(),
                                });

                                let modified_array_exprs: Vec<_> = (0..element_count)
                                    .map(|i| {
                                        let array_access = Expr::Index(ExprIndex {
                                            attrs: vec![],
                                            expr: Box::new(matrix_to_array_expr.clone()),
                                            bracket_token: Default::default(),
                                            index: Box::new(Expr::Lit(ExprLit {
                                                attrs: vec![],
                                                lit: syn::Lit::Int(syn::LitInt::new(
                                                    &i.to_string(),
                                                    Span::call_site(),
                                                )),
                                            })),
                                        });

                                        let func_call = Expr::MethodCall(ExprMethodCall {
                                            attrs: vec![],
                                            receiver: Box::new(array_access),
                                            method: Ident::new(func_name, Span::call_site()),
                                            turbofish: None,
                                            dot_token: token::Dot::default(),
                                            paren_token: token::Paren::default(),
                                            args: Punctuated::from_iter(
                                                args_exprs.iter().cloned().map(|(expr, _)| expr),
                                            ),
                                        });

                                        Expr::Group(ExprGroup {
                                            attrs: vec![],
                                            group_token: token::Group::default(),
                                            expr: Box::new(func_call),
                                        })
                                    })
                                    .collect();

                                Expr::Call(ExprCall {
                                    attrs: vec![],
                                    func: Box::new(Expr::Path(ExprPath {
                                        attrs: vec![],
                                        qself: None,
                                        path: Path::from(Ident::new(
                                            "from_cols_array",
                                            Span::call_site(),
                                        )),
                                    })),
                                    paren_token: token::Paren::default(),
                                    args: Punctuated::from_iter(vec![Expr::Array(ExprArray {
                                        attrs: vec![],
                                        bracket_token: Default::default(),
                                        elems: Punctuated::from_iter(modified_array_exprs),
                                    })]),
                                })
                            }
                            _ => {
                                // General case for other functions
                                todo!()
                            }
                        }
                    }
                    _ => todo!(),
                };
                Ok(expr)
            }
            x => todo!("{x:?}"),
        }
    }

    fn convert_infinite_loop(&mut self, body_stmts: Vec<Stmt>) -> Result<Stmt, WriterError> {
        log::trace!("Infinte loop");
        Ok(Stmt::Expr(
            Expr::Loop(syn::ExprLoop {
                attrs: vec![],
                label: None,
                loop_token: Default::default(),
                body: SynBlock {
                    brace_token: Default::default(),
                    stmts: body_stmts,
                },
            }),
            None,
        ))
    }

    fn convert_conditional_break_loop(
        &mut self,
        condition_expr: Expr,
        body_stmts: Vec<Stmt>,
    ) -> Result<Stmt, WriterError> {
        log::trace!("Conditional break loop");
        // Create a conditional break statement
        let break_stmt = Stmt::Expr(
            Expr::If(ExprIf {
                attrs: vec![],
                if_token: Default::default(),
                cond: Box::new(Expr::Unary(ExprUnary {
                    attrs: vec![],
                    op: syn::UnOp::Not(token::Not::default()),
                    expr: Box::new(Expr::Paren(ExprParen {
                        attrs: vec![],
                        paren_token: token::Paren::default(),
                        expr: Box::new(condition_expr),
                    })),
                })),
                then_branch: SynBlock {
                    brace_token: Default::default(),
                    stmts: vec![Stmt::Expr(
                        Expr::Break(syn::ExprBreak {
                            attrs: vec![],
                            break_token: Default::default(),
                            label: None,
                            expr: None,
                        }),
                        Some(token::Semi::default()),
                    )],
                },
                else_branch: None,
            }),
            None,
        );

        let mut new_body_stmts = vec![break_stmt];
        new_body_stmts.extend(body_stmts);

        self.convert_infinite_loop(new_body_stmts)
    }

    fn convert_type(
        &mut self,
        types: &UniqueArena<Type>,
        ty: &Handle<Type>,
    ) -> Result<syn::Type, WriterError> {
        let ty = &types[*ty];
        log::trace!("Converting type: {ty:#?}");
        match &ty.inner {
            &TypeInner::Scalar(scalar @ Scalar { kind, width }) => match (kind, width) {
                (ScalarKind::Sint, 1) => Ok(syn::Type::Path(syn::TypePath {
                    qself: None,
                    path: Path::from(Ident::new("i8", Span::call_site())),
                })),
                (ScalarKind::Sint, 2) => Ok(syn::Type::Path(syn::TypePath {
                    qself: None,
                    path: Path::from(Ident::new("i16", Span::call_site())),
                })),
                (ScalarKind::Sint, 4) => Ok(syn::Type::Path(syn::TypePath {
                    qself: None,
                    path: Path::from(Ident::new("i32", Span::call_site())),
                })),
                (ScalarKind::Sint, 8) => Ok(syn::Type::Path(syn::TypePath {
                    qself: None,
                    path: Path::from(Ident::new("i64", Span::call_site())),
                })),
                (ScalarKind::Uint, 1) => Ok(syn::Type::Path(syn::TypePath {
                    qself: None,
                    path: Path::from(Ident::new("u8", Span::call_site())),
                })),
                (ScalarKind::Uint, 2) => Ok(syn::Type::Path(syn::TypePath {
                    qself: None,
                    path: Path::from(Ident::new("u16", Span::call_site())),
                })),
                (ScalarKind::Uint, 4) => Ok(syn::Type::Path(syn::TypePath {
                    qself: None,
                    path: Path::from(Ident::new("u32", Span::call_site())),
                })),
                (ScalarKind::Uint, 8) => Ok(syn::Type::Path(syn::TypePath {
                    qself: None,
                    path: Path::from(Ident::new("u64", Span::call_site())),
                })),
                (ScalarKind::Float, 4) => Ok(syn::Type::Path(syn::TypePath {
                    qself: None,
                    path: Path::from(Ident::new("f32", Span::call_site())),
                })),
                (ScalarKind::Float, 8) => Ok(syn::Type::Path(syn::TypePath {
                    qself: None,
                    path: Path::from(Ident::new("f64", Span::call_site())),
                })),
                (ScalarKind::Bool, _) => Ok(syn::Type::Path(syn::TypePath {
                    qself: None,
                    path: Path::from(Ident::new("bool", Span::call_site())),
                })),
                _ => Err(WriterError::UnsupportedScalarType(scalar)),
            },
            &TypeInner::Vector { .. } => {
                let vec_type = map_type_to_glam(ty)?;
                let segments = vec![
                    PathSegment::from(Ident::new("glam", Span::call_site())),
                    PathSegment::from(Ident::new(&vec_type, Span::call_site())),
                ];
                let path = Path {
                    leading_colon: None,
                    segments: Punctuated::from_iter(segments),
                };

                Ok(syn::Type::Path(syn::TypePath { qself: None, path }))
            }
            &TypeInner::Matrix { .. } => {
                let mat_type = map_type_to_glam(ty)?;

                let segments = vec![
                    PathSegment::from(Ident::new("glam", Span::call_site())),
                    PathSegment::from(Ident::new(&mat_type, Span::call_site())),
                ];
                let path = Path {
                    leading_colon: None,
                    segments: Punctuated::from_iter(segments),
                };

                Ok(syn::Type::Path(syn::TypePath { qself: None, path }))
            }
            &TypeInner::Struct { ref members, .. } => {
                if let Some(struct_name) = &ty.name {
                    let ident = syn::Ident::new(&struct_name, Span::call_site());

                    // Create a path for the struct type
                    let path = Path {
                        leading_colon: None,
                        segments: Punctuated::from_iter(vec![PathSegment::from(ident)]),
                    };

                    Ok(syn::Type::Path(syn::TypePath { qself: None, path }))
                } else {
                    assert_eq!(members.len(), 1);
                    let handle = members.first().expect("exactly one struct member").ty;
                    self.convert_type(types, &handle)
                }
            }
            x => todo!("Handle type {x:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::front::glsl::{Frontend, Options};
    use crate::valid::{Capabilities, ValidationFlags, Validator};
    use crate::ShaderStage;

    fn assert_correct_translation(stage: ShaderStage, glsl_code: &str, rust_code: &str) {
        let mut frontend = Frontend::default();
        let options = Options::from(stage);
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        let module = frontend
            .parse(&options, glsl_code)
            .expect("glsl input parses");
        let info = validator.validate(&module).expect("valid glsl input");

        let mut writer = Writer::new(Target::Cpu, WriterFlags::empty());
        let generated = prettyplease::unparse(&writer.write_module(&module, &info).unwrap());
        let expected = prettyplease::unparse(&syn::parse_file(rust_code).unwrap());
        assert_eq!(generated, expected);
    }

    #[test]
    fn test_simple_shader_translation() {
        let glsl = r#"
            #version 450
            void main() {
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main() {
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_var_decl() {
        let glsl = r#"
            void main() {
                int a;
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut a: i32;
            }
        "#;
        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_simple_assignments() {
        let glsl = r#"
            void main() {
                int a = 42;
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut a: i32 = 42;
            }
        "#;
        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_deferred_assignment() {
        let glsl = r#"
            void main() {
                int a;
                a = 42;
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut a: i32;
                a = 42;
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_function_with_assignments() {
        let glsl = r#"
            #version 450
            void main() {
                int a;
                a = 5;
                int b;
                b = a;
            }
        "#;

        // TODO: make order match.
        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut a: i32;
                let mut b: i32;
                a = 5;
                b = a;
            }
        "#;
        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_compound_ops() {
        let glsl = r#"
            #version 450
            void main() {
                int a;
                int b;
                a += 1;
                a = b + 1;
                a = a + b + 1;
            }
        "#;

        // TODO: make order match.
        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut a: i32;
                let mut b: i32;
                a += 1;
                a = b + 1;
                a = a + b + 1;
            }
        "#;
        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_if() {
        let glsl = r#"
            #version 450
            void main() {
                int a = 1;
                int b = 2;
                if ( a < b ) {
                    a = 3;
                }
            }
        "#;

        // TODO: make order match.
        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut a: i32 = 1;
                let mut b: i32 = 2;
                if a < b {
                    a = 3;
                }
            }
        "#;
        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_else() {
        let glsl = r#"
            #version 450
            void main() {
                int a = 1;
                int b = 2;
                if ( a < b ) {
                    a = 3;
                } else {
                    a = 4;
                    b = 5;
                }
            }
        "#;

        // TODO: make order match.
        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut a: i32 = 1;
                let mut b: i32 = 2;
                if a < b {
                    a = 3;
                } else {
                    a = 4;
                    b = 5;
                }
            }
        "#;
        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_while_loop() {
        let glsl = r#"
            void main() {
                int a = 0;
                while (a < 10) {
                    a += 1;
                }
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut a: i32 = 0;
                while a < 10 {
                    a += 1;
                }
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_for_loop() {
        let glsl = r#"
            void main() {
                int a = 0;
                for (int x = 0; x < 100; x++) {
                    a = x;
                }
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut a: i32 = 0;
                let mut x: i32 = 0;
                while x < 100 {
                    a = x;
                }
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_do_while_loop() {
        let glsl = r#"
            void main() {
                int a = 0;
                do {
                  a += 1;
                } while (a >= 5);
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut a: i32 = 0;
                loop {
                    a += 1;
                    if !(a >= 5) {
                        break;
                    }
                }
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_vector_and_matrix_definitions_with_scalars() {
        let glsl = r#"
            #version 450
            void main() {
                vec2 a = vec2(1.0, 2.0);
                vec3 b = vec3(3.0, 4.0, 5.0);
                mat2 c = mat2(1.0, 2.0, 3.0, 4.0);
                mat3 d = mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
                mat4 e = mat4(1.0); // identity matrix

                ivec2 f = ivec2(1, 2);
                ivec3 g = ivec3(3, 4, 5);
                uvec2 h = uvec2(1u, 2u);
                uvec3 i = uvec3(3u, 4u, 5u);
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut a: glam::Vec2 = glam::Vec2::new(1.0, 2.0);
                let mut b: glam::Vec3 = glam::Vec3::new(3.0, 4.0, 5.0);
                let mut c: glam::Mat2 = glam::Mat2::from_cols(
                    glam::Vec2::new(1.0, 2.0),
                    glam::Vec2::new(3.0, 4.0),
                );
                let mut d: glam::Mat3 = glam::Mat3::from_cols(
                    glam::Vec3::new(1.0, 0.0, 0.0),
                    glam::Vec3::new(0.0, 1.0, 0.0),
                    glam::Vec3::new(0.0, 0.0, 1.0),
                );
                let mut e: glam::Mat4 = glam::Mat4::from_cols(
                    glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
                    glam::Vec4::new(0.0, 1.0, 0.0, 0.0),
                    glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
                    glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
                );
                let mut f: glam::IVec2 = glam::IVec2::new(1, 2);
                let mut g: glam::IVec3 = glam::IVec3::new(3, 4, 5);
                let mut h: glam::UVec2 = glam::UVec2::new(1, 2);
                let mut i: glam::UVec3 = glam::UVec3::new(3, 4, 5);
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_continue_in_loop() {
        let glsl = r#"
            void main() {
                for (int i = 0; i < 10; i++) {
                    if (i < 5) continue;
                }
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut i: i32 = 0;
                while i < 10 {
                    if i < 5 {
                        continue;
                    }
                }
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_splat_translation() {
        let glsl = r#"
            void main() {
                vec4 color = vec4(1.0);
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut color: glam::Vec4 = glam::Vec4::splat(1.0);
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_math_expressions_translation() {
        let glsl = r#"
            void main() {
                float a = 1.0;
                float b = 2.0;
                float c = a + b;
                float d = sin(a);
                float e = pow(a, b);

                vec2 v1 = vec2(1.0, 2.0);
                vec2 v2 = vec2(3.0, 4.0);
                vec2 v3 = v1 + v2;
                vec2 v4 = sin(v1);
                vec2 v5 = pow(v1, v2);
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut a: f32 = 1.0;
                let mut b: f32 = 2.0;
                let mut c: f32;
                let mut d: f32;
                let mut e: f32;
                let mut v1: glam::Vec2 = glam::Vec2::new(1.0, 2.0);
                let mut v2: glam::Vec2 = glam::Vec2::new(3.0, 4.0);
                let mut v3: glam::Vec2;
                let mut v4: glam::Vec2;
                let mut v5: glam::Vec2;
                c = a + b;
                d = a.sin();
                e = a.powf(b);
                v3 = v1 + v2;
                v4 = glam::Vec2::new(v1.x.sin(), v1.y.sin());
                v5 = glam::Vec2::new(v1.x.powf(v2.x), v1.y.powf(v2.y));
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_swizzling() {
        let glsl = r#"
            #version 450
            void main() {
                vec4 vec = vec4(1.0, 2.0, 3.0, 4.0);
                float x = vec.x;
                vec2 xy = vec.xy;
                vec3 xyz = vec.xyz;
                vec.w = 5.0;
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut vec: glam::Vec4 = glam::Vec4::new(1.0, 2.0, 3.0, 4.0);
                let mut x: f32;
                let mut xy: glam::Vec2;
                let mut xyz: glam::Vec3;
                x = vec.x;
                xy = vec.xy();
                xyz = vec.xyz();
                vec.w = 5.0;
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_vector_and_matrix_indexing() {
        let glsl = r#"
            #version 450
            void main() {
                vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
                float x = v[0];
                float y = v[1];

                mat2 m = mat2(1.0, 0.0, 0.0, 1.0);
                vec2 col0 = m[0];
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut v: glam::Vec4 = glam::Vec4::new(1.0, 2.0, 3.0, 4.0);
                let mut x: f32;
                let mut y: f32;
                let mut m: glam::Mat2 = glam::Mat2::from_cols(
                    glam::Vec2::new(1.0, 0.0),
                    glam::Vec2::new(0.0, 1.0),
                );
                let mut col0: glam::Vec2;
                x = v.x;
                y = v.y;
                col0 = m.x_axis;
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_constant_translation() {
        let glsl = r#"
            const int MY_CONSTANT = 5;
            void main() {
                int a = MY_CONSTANT;
                a += 1;
            }
        "#;

        let rust = r#"
            const MY_CONSTANT: i32 = 5;

            #[spirv(fragment)]
            fn main() {
                let mut a: i32 = MY_CONSTANT;
                a += 1;
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    #[cfg(FAILING)]
    fn test_function_args() {
        let glsl = r#"
            vec4 mixColors(vec4 color1, vec4 color2, float factor) {
                return mix(color1, color2, factor);
            }
            
            void main() {
                vec4 color1 = vec4(1.0, 0.0, 0.0, 1.0); // Red color
                vec4 color2 = vec4(0.0, 0.0, 1.0, 1.0); // Blue color
                float mixFactor = 0.5; // 50% mix
            
                // Use the function to mix colors
                vec4 mixed = mixColors(color1, color2, mixFactor);
            }
        "#;

        // TODO: Keep same order.
        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut color1: glam::Vec4 = glam::Vec4::new(1.0, 0.0, 0.0, 1.0);
                let mut color2: glam::Vec4 = glam::Vec4::new(0.0, 0.0, 1.0, 1.0);
                let mut mixFactor: f32 = 0.5;
                let mut mixed: glam::Vec4;
                mixed = mixColors(color1, color2, factor);
            }

            fn mixColors(color1: glam::Vec4, color2: glam::Vec4, factor: f32) -> glam::Vec4 {
                let mut color1_1: glam::Vec4;
                let mut color2_1: glam::Vec4;
                let mut factor_1: f32;
                color1_1 = color1;
                color2_1 = color2;
                factor_1 = factor;
                return color1_1.lerp(color2_1, factor_1);
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_mix_with_scalar() {
        let glsl = r#"
            void main() {
                vec3 color1 = vec3(1.0, 0.0, 0.0);
                vec3 color2 = vec3(0.0, 1.0, 0.0);
                float factor = 0.5;
                vec3 result = mix(color1, color2, factor);
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut color1: glam::Vec3 = glam::Vec3::new(1.0, 0.0, 0.0);
                let mut color2: glam::Vec3 = glam::Vec3::new(0.0, 1.0, 0.0);
                let mut factor: f32 = 0.5;
                let mut result: glam::Vec3;
                result = color1.lerp(color2, factor);
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_mix_with_vec3() {
        let glsl = r#"
            void main() {
                vec3 color1 = vec3(1.0, 0.0, 0.0);
                vec3 color2 = vec3(0.0, 1.0, 0.0);
                vec3 factor = vec3(0.5, 0.5, 0.5);
                vec3 result = mix(color1, color2, factor);
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main() {
                let mut color1: glam::Vec3 = glam::Vec3::new(1.0, 0.0, 0.0);
                let mut color2: glam::Vec3 = glam::Vec3::new(0.0, 1.0, 0.0);
                let mut factor: glam::Vec3 = glam::Vec3::new(0.5, 0.5, 0.5);
                let mut result: glam::Vec3;
                result = glam::Vec3::new(
                    color1.x.lerp(color2.x, factor.x),
                    color1.y.lerp(color2.y, factor.y),
                    color1.z.lerp(color2.z, factor.z),
                );
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_shader_entrypoint_translation() {
        let glsl = r#"
            #version 450
            void main() {
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main() {
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_location_binding_translation() {
        let glsl = r#"
            #version 450
            layout(location = 0) in vec4 color;
            layout(location = 1) in vec2 blah;
            void main() {
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main(color: glam::Vec4, #[spirv(location = 1)] blah: glam::Vec2) {
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }

    #[test]
    fn test_location_binding_with_sampling() {
        let glsl = r#"
            #version 450
            layout(location = 1) centroid in vec4 color;
            void main() {
            }
        "#;

        let rust = r#"
            #[spirv(fragment)]
            fn main(#[spirv(location = 1, centroid)] color: glam::Vec4) {
            }
        "#;

        assert_correct_translation(ShaderStage::Fragment, glsl, rust);
    }
}
