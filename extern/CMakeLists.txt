include(FetchContent)

FetchContent_Declare(
  knncolle 
  GIT_REPOSITORY https://github.com/LTLA/knncolle
  GIT_TAG 384cbc61f65d89817ac45ad534e380fd8ac08a3f
)

FetchContent_MakeAvailable(knncolle)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/LTLA/tatami
  GIT_TAG b2741f25b88dfb3a6a9e9cd8d9b38550ff12c5e0
)

FetchContent_MakeAvailable(tatami)
