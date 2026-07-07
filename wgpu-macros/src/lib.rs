use heck::ToSnakeCase;
use proc_macro::TokenStream;
use quote::quote;
use syn::Ident;

/// Creates a test that will run on all gpus on a given system.
///
/// Apply this macro to a static variable with a type that can be converted to a `GpuTestConfiguration`.
#[proc_macro_attribute]
pub fn gpu_test(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_static = syn::parse_macro_input!(item as syn::ItemStatic);
    let vis = input_static.vis;
    let expr = &input_static.expr;
    let ident = &input_static.ident;
    let ident_str = ident.to_string();
    let ident_lower = ident_str.to_snake_case();

    let register_test_name = Ident::new(&format!("{ident}"), ident.span());

    quote! {
        #[allow(non_snake_case)]
        #vis fn #register_test_name() -> ::wgpu_test::GpuTestConfiguration {
            struct S;

            // Allow any type that can be converted to a GpuTestConfiguration
            ::wgpu_test::GpuTestConfiguration::from(#expr).name_from_init_function_typename::<S>(#ident_lower)
        }
    }
    .into()
}
