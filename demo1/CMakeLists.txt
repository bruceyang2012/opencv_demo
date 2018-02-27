project(demo1)

find_package(Threads REQUIRED)
if(THREADS_HAVE_PTHREAD_ARG)
  set_property(TARGET thread_demo.cpp PROPERTY COMPILE_OPTIONS "-pthread")
  set_property(TARGET thread_demo.cpp PROPERTY INTERFACE_COMPILE_OPTIONS "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
  target_link_libraries(thread_demo.cpp "${CMAKE_THREAD_LIBS_INIT}")
endif()

add_executable(demo1 thread_demo.cpp)
