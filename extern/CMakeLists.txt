include(FetchContent)

FetchContent_Declare(
  subpar
  GIT_REPOSITORY https://github.com/LTLA/subpar
  GIT_TAG master
)


FetchContent_Declare(
  knncolle 
  GIT_REPOSITORY https://github.com/knncolle/knncolle
  GIT_TAG master
)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/tatami-inc/tatami
  GIT_TAG master
)

FetchContent_MakeAvailable(subpar)
FetchContent_MakeAvailable(knncolle)
FetchContent_MakeAvailable(tatami)
