# add_executable(sparseflow ${PROJ_SRC_FILES} ${SFLOW_SRC_FILES} "sparseflow.cpp")
# target_link_libraries(sparseflow ${OpenCV_LIBS} ${Boost_LIBRARIES} kitti viso2)
# install(TARGETS sparseflow DESTINATION bin)

pybind11_add_module(pyinterface pyinterface.cpp)
target_link_libraries(pyinterface PUBLIC ${Boost_LIBRARIES} viso2)