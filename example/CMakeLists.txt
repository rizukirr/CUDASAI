function(add_cudasai_example name)
  add_executable(example_${name} ${name}.cu helper.cu)
  target_link_libraries(example_${name} PRIVATE cdscore)
endfunction()

add_cudasai_example(linear)
add_cudasai_example(add)
