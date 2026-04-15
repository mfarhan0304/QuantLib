find_package(OpenMP)

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

add_executable(PortfolioVaRValidate
    validate.cpp
    portfolio.cpp
    scenarios.cpp
)
target_link_libraries(PortfolioVaRValidate ql_library ${QL_THREAD_LIBRARIES})

# ql_library references OMP symbols (gaussian1d engines etc.) so all targets
# that link it need libgomp, not just PortfolioVaROMP.
if(OpenMP_CXX_FOUND)
    foreach(tgt PortfolioVaR PortfolioVaROMP PortfolioVaRValidate)
        target_link_libraries(${tgt} OpenMP::OpenMP_CXX)
    endforeach()
    # Only PortfolioVaROMP gets the compile flag (-fopenmp) since it
    # actively uses #pragma omp directives.
    target_compile_options(PortfolioVaROMP PRIVATE ${OpenMP_CXX_FLAGS})
endif()

if(QL_INSTALL_EXAMPLES)
    install(TARGETS PortfolioVaR PortfolioVaROMP PortfolioVaRValidate
            RUNTIME DESTINATION ${QL_INSTALL_EXAMPLESDIR})
endif()
