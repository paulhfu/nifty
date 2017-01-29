find_package(VIGRA)

add_executable(test_batch_prediction test_batch_prediction.cxx)
target_link_libraries(test_batch_prediction ${TEST_LIBS} ${TBB_LIBRARIES} ${FASTFILTERS_LIBRARY}
        ${HDF5_LIBRARIES} ${VIGRA_IMPEX_LIBRARY})
add_test(test_batch_prediction test_batch_prediction)

#add_executable(test_interactive_classification test_interactive_classification.cxx)
#target_link_libraries(test_interactive_classification ${TEST_LIBS} ${TBB_LIBRARIES} ${FASTFILTERS_LIBRARY}
#        ${HDF5_LIBRARIES} ${VIGRA_IMPEX_LIBRARY})
#add_test(test_interactive_classification test_interactive_classification)

#add_executable(test_load_rf test_load_rf.cxx)
#target_link_libraries(test_load_rf ${TEST_LIBS} ${TBB_LIBRARIES} ${FASTFILTERS_LIBRARY}
#        ${HDF5_LIBRARIES} ${VIGRA_IMPEX_LIBRARY})
#add_test(test_load_rf test_load_rf)
