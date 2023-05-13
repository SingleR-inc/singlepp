include(FetchContent)

FetchContent_Declare(
  knncolle 
  GIT_REPOSITORY https://github.com/LTLA/knncolle
  GIT_TAG c5a1776ebf10641d9bf8715e89cb2d965b06e899
)

FetchContent_MakeAvailable(knncolle)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/LTLA/tatami
  GIT_TAG e2097fcaa877b6fa8b642a73deda8daaffadc209
)

FetchContent_MakeAvailable(tatami)

FetchContent_Declare(
  byteme
  GIT_REPOSITORY https://github.com/LTLA/byteme
  GIT_TAG c638112d307eb873ec0c719046cf202d061dc4d0
)

FetchContent_MakeAvailable(byteme)
