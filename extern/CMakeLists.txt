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
  GIT_REPOSITORY https://github.com/LTLA/byteme
  GIT_TAG 2e2185a52dd474d28cab89eb79329b467b8dac8e
)

FetchContent_MakeAvailable(byteme)
