# Following: https://cliutils.gitlab.io/modern-cmake/chapters/testing/googletest.html

add_subdirectory("${PROJECT_SOURCE_DIR}/googletest" "googletest")

mark_as_advanced(
        BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
        gmock_build_tests gtest_build_samples gtest_build_tests
        gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)

set_target_properties(gtest PROPERTIES FOLDER extern)
set_target_properties(gtest_main PROPERTIES FOLDER extern)
set_target_properties(gmock PROPERTIES FOLDER extern)
set_target_properties(gmock_main PROPERTIES FOLDER extern)

macro(package_add_test TESTNAME)

    # create an exectuable in which the tests will be stored
    if (ENABLE_CUDA)
        cuda_add_executable(${TESTNAME} ${ARGN})
    else()
        add_executable(${TESTNAME} ${ARGN})
    endif()

    target_include_directories(${TESTNAME} PUBLIC ${PROJECT_SOURCE_DIR}/src)

    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove g_test_main if writing your own main function.
    target_link_libraries(${TESTNAME} gtest gmock gtest_main ${LINK_LIBRARIES})
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
            # set a working directory so your project root so that you can find test data via paths relative to the project root
            WORKING_DIRECTORY ${PROJECT_DIR}
            PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
            )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

package_add_test(test_regular_grid_loading test_data_loading.cpp)
package_add_test(test_mese test_mese.cpp)
package_add_test(test_moment_image test_moment_image.cpp)
package_add_test(test_prediction_coding test_prediction_coding.cpp)
package_add_test(test_moment_compaction test_moment_compaction.cpp)
package_add_test(test_moment_image_interpolator test_moment_image_interpolator.cpp)
package_add_test(test_rayhistogram test_rayhistogram.cpp)
package_add_test(test_moment_image_coding test_moment_image_coding.cpp)
package_add_test(test_coding_transform test_coding_transform.cpp)
package_add_test(test_error_measures test_error_measures.cpp)