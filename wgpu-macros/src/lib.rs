use heck::ToSnakeCase;
use proc_macro::TokenStream;
use quote::quote;
use syn::Ident;

#[proc_macro_attribute]
pub fn gpu_test(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_static = syn::parse_macro_input!(item as syn::ItemStatic);
    let expr = &input_static.expr;
    let ident = &input_static.ident;
    let ident_str = ident.to_string();
    let ident_lower = ident_str.to_snake_case();

    let register_test_name = Ident::new(&format!("{}_initializer", ident_lower), ident.span());
    let test_name_webgl = Ident::new(&format!("{}_webgl", ident_lower), ident.span());

    quote! {
        #[cfg(not(target_arch = "wasm32"))]
        #[::wgpu_test::ctor]
        fn #register_test_name() {
            ::wgpu_test::infra::TEST_LIST.lock().push(
                // Allow any type that can be converted to a GpuTestConfiguration
                ::wgpu_test::infra::GpuTestConfiguration::from(#expr).name_if_not_set(#ident_lower)
            )
        }

        #[cfg(target_arch = "wasm32")]
        fn #test_name_webgl() {
            // todo
        }
    }
    .into()
}
