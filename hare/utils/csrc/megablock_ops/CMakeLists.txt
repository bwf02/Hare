pybind11_add_module(megablocks_ops MODULE
        ops.cu
)
target_include_directories(megablocks_ops PUBLIC
        ${MEGABLOCKS_OPS_SRC_DIR}
        ${LIBTORCH}/include
)
target_link_libraries(megablocks_ops 
        PRIVATE "${TORCH_LIBRARIES}"
        PRIVATE ${TORCH_PYTHON_LIBRARY} #没有会导致ImportError: undefined symbol: _ZN8pybind116detail11type_casterIN2at6TensorEvE4loadENS_6handleEb
)
