




# Create the binding library
pybind11_add_module(_region_growing 
    region_growing.cxx
    edge_based_watershed.cxx
)

# Don't add a 'lib' prefix to the shared library
set_target_properties(_region_growing PROPERTIES PREFIX "")

# Link against the Python shared library
target_link_libraries(_region_growing ${PYTHON_LIBRARIES})




file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/nifty/region_growing)

ADD_CUSTOM_COMMAND(
        TARGET _region_growing
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different $<TARGET_FILE:_region_growing> ${CMAKE_BINARY_DIR}/python/nifty/region_growing
        COMMENT "Copying target _region_growing to temporary module directory")




add_custom_target(copy_python_region_growing__init__ ALL#POST_BUILD
    COMMAND cmake -E copy_if_different 
    ${CMAKE_SOURCE_DIR}/src/python/module/region_growing/__init__.py 
    ${CMAKE_BINARY_DIR}/python/nifty/region_growing/__init__.py)




add_dependencies(python-module _region_growing)
add_dependencies(python-module copy_python_region_growing__init__)
