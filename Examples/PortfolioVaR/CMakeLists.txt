add_executable(PortfolioVaR
    PortfolioVaR.cpp
    portfolio.cpp
    scenarios.cpp
)
target_link_libraries(PortfolioVaR ql_library ${QL_THREAD_LIBRARIES})

add_executable(PortfolioVaROMP
    PortfolioVaROMP.cpp
    portfolio.cpp
    scenarios.cpp
)
target_link_libraries(PortfolioVaROMP ql_library ${QL_THREAD_LIBRARIES})
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(PortfolioVaROMP OpenMP::OpenMP_CXX)
    target_compile_options(PortfolioVaROMP PRIVATE ${OpenMP_CXX_FLAGS})
endif()

add_executable(PortfolioVaRValidate
    validate.cpp
    portfolio.cpp
    scenarios.cpp
)
target_link_libraries(PortfolioVaRValidate ql_library ${QL_THREAD_LIBRARIES})

if(QL_INSTALL_EXAMPLES)
    install(TARGETS PortfolioVaR PortfolioVaROMP PortfolioVaRValidate
            RUNTIME DESTINATION ${QL_INSTALL_EXAMPLESDIR})
endif()
