include(FetchContent)

FetchContent_Declare(
  knncolle 
  GIT_REPOSITORY https://github.com/LTLA/knncolle
  GIT_TAG be41b6ef97bd522d651455755efd353abc0c40b2
)

FetchContent_MakeAvailable(knncolle)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/LTLA/tatami
  GIT_TAG b2741f25b88dfb3a6a9e9cd8d9b38550ff12c5e0
)

FetchContent_MakeAvailable(tatami)

FetchContent_Declare(
  byteme
  GIT_REPOSITORY https://github.com/clusterfork/buffin
  GIT_TAG 55730cfa3b01eb4d741e99b4069ea99ef2cf359f
)

FetchContent_MakeAvailable(byteme)
