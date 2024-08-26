include(FetchContent)

FetchContent_Declare(
  subpar
  GIT_REPOSITORY https://github.com/LTLA/subpar
  GIT_TAG master  # ^0.2.0
)

FetchContent_Declare(
  knncolle 
  GIT_REPOSITORY https://github.com/knncolle/knncolle
  GIT_TAG master  # ^2.0.0
)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/tatami-inc/tatami
  GIT_TAG master # ^3.0.0
)

FetchContent_MakeAvailable(subpar)
FetchContent_MakeAvailable(knncolle)
FetchContent_MakeAvailable(tatami)
