pybind11_add_module(hare_ops MODULE
    ops.cu
)

target_include_directories(hare_ops PUBLIC
        ${HARE_OPS_SRC_DIR}
        ${LIBTORCH}/include
)
target_link_libraries(hare_ops 
        PRIVATE "${TORCH_LIBRARIES}"
        PRIVATE ${TORCH_PYTHON_LIBRARY} #没有会导致ImportError: undefined symbol: _ZN8pybind116detail11type_casterIN2at6TensorEvE4loadENS_6handleEb
)
