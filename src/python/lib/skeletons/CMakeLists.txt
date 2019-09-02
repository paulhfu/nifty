
addPythonModule(
    NESTED_NAME 
        nifty/skeletons
    SOURCES
        skeletons.cxx
        evaluation.cxx
        helper.cxx
    LIBRRARIES
        ${Z5_COMPRESSION_LIBRARIES}
        ${FILESYSTEM_LIBRARIES}    
        Threads::Threads
)
