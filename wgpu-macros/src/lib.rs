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
    let expr = &input_static.expr;
    let ident = &input_static.ident;
    let ident_str = ident.to_string();
    let ident_lower = ident_str.to_snake_case();

    let register_test_name = Ident::new(&format!("{ident_lower}_initializer"), ident.span());
    let test_name_webgl = Ident::new(&format!("{ident_lower}_webgl"), ident.span());

    quote! {
        #[cfg(not(target_arch = "wasm32"))]
        #[::wgpu_test::ctor]
        fn #register_test_name() {
            struct S;

            ::wgpu_test::native::TEST_LIST.lock().push(
                // Allow any type that can be converted to a GpuTestConfiguration
                ::wgpu_test::GpuTestConfiguration::from(#expr).name_from_init_function_typename::<S>(#ident_lower)
            )
        }

        #[cfg(target_arch = "wasm32")]
        #[wasm_bindgen_test::wasm_bindgen_test]
        async fn #test_name_webgl() {
            struct S;

            // Allow any type that can be converted to a GpuTestConfiguration
            let test_config = ::wgpu_test::GpuTestConfiguration::from(#expr).name_from_init_function_typename::<S>(#ident_lower);

            ::wgpu_test::execute_test(None, test_config, None).await;
        }
    }
    .into()
}
