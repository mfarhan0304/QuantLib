add_executable(BermudanSwaption BermudanSwaption.cpp)
target_link_libraries(BermudanSwaption ql_library ${QL_THREAD_LIBRARIES})
<<<<<<< HEAD

add_executable(BermudanSwaptionOMP BermudanSwaptionOMP.cpp)
target_link_libraries(BermudanSwaptionOMP ql_library ${QL_THREAD_LIBRARIES})

if (QL_INSTALL_EXAMPLES)
    install(TARGETS BermudanSwaption BermudanSwaptionOMP
            RUNTIME DESTINATION ${QL_INSTALL_EXAMPLESDIR})
=======
if (QL_INSTALL_EXAMPLES)
    install(TARGETS BermudanSwaption RUNTIME DESTINATION ${QL_INSTALL_EXAMPLESDIR})
>>>>>>> 8aef029c02935baf52c93391eb70dcdbd9ab88aa
endif()

