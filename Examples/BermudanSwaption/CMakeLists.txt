add_executable(BermudanSwaption BermudanSwaption.cpp)
target_link_libraries(BermudanSwaption ql_library ${QL_THREAD_LIBRARIES})

add_executable(BermudanSwaptionOMP BermudanSwaptionOMP.cpp)
target_link_libraries(BermudanSwaptionOMP ql_library ${QL_THREAD_LIBRARIES})

if (QL_INSTALL_EXAMPLES)
    install(TARGETS BermudanSwaption BermudanSwaptionOMP
            RUNTIME DESTINATION ${QL_INSTALL_EXAMPLESDIR})
endif()

