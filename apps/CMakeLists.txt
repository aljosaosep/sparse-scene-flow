add_executable(sparseflow ${PROJ_SRC_FILES} ${SUN_UTILS_SRC_FILES} "sparseflow.cpp")
target_link_libraries(sparseflow ${OpenCV_LIBS} ${Boost_LIBRARIES}  viso2)
install(TARGETS sparseflow DESTINATION bin)
