include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/tatami-inc/tatami
  GIT_TAG master # ^4.0.0
)

FetchContent_Declare(
  aarand
  GIT_REPOSITORY https://github.com/LTLA/aarand
  GIT_TAG master # ^1.1.0
)

FetchContent_Declare(
  sanisizer
  GIT_REPOSITORY https://github.com/LTLA/sanisizer
  GIT_TAG master # ^0.1.5
)

FetchContent_MakeAvailable(tatami)
FetchContent_MakeAvailable(aarand)
FetchContent_MakeAvailable(sanisizer)
