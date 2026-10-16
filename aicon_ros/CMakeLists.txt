cmake_minimum_required(VERSION 3.8)
project(aicon_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ament_lint_auto)

rosidl_generate_interfaces(${PROJECT_NAME}
   "msg/QuantityMsg.msg"
   "msg/StringListMsg.msg"
   "msg/ListOfStringListsMsg.msg"
   "msg/ListOfListOfStringListsMsg.msg"
   "msg/TensorListMsg.msg"
   "msg/DerivativesMsg.msg"
   "msg/QuantityAndForwardDerivativesMsg.msg"
   DEPENDENCIES std_msgs
 )


# add_dependencies(${py_typesupport_target} ${cpp_typesupport_target})

 # Python install
ament_python_install_package(aicon_ros_python_wrapping)

# meshes used by the RViz markers (package://aicon_ros/meshes/...)
install(DIRECTORY meshes DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
