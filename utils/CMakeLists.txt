cmake_minimum_required(VERSION 3.15)

project(rknn_model_zoo_utils)

add_library(fileutils STATIC
    file_utils.c
)
target_include_directories(fileutils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(imagedrawing STATIC
    image_drawing.c
)
target_include_directories(imagedrawing PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)



add_definitions(-DLIBRGA_IM2D_HANDLE)

add_library(imageutils STATIC
    image_utils.c
)

target_include_directories(imageutils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${STB_INCLUDES}
    ${LIBRGA_INCLUDES}
)

target_link_libraries(imageutils
    ${LIBRGA}
)
# 如果需要禁用 libjpeg
set(DISABLE_LIBJPEG ON)


if (DISABLE_LIBJPEG)
    add_definitions(-DDISABLE_LIBJPEG)
else()
    target_link_libraries(imageutils
        ${LIBJPEG}
    )
    target_include_directories(imageutils PUBLIC
        ${LIBJPEG_INCLUDES}
    )
endif()

add_library(audioutils STATIC
    audio_utils.c
)

target_link_libraries(audioutils
    ${LIBSNDFILE}
)

target_include_directories(audioutils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBSNDFILE_INCLUDES}
)
