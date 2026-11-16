# Sanitizer support, selected by the ANNEX_SANITIZE cache variable.
#
#   ""        no sanitizer
#   "address" AddressSanitizer + UndefinedBehaviorSanitizer (they combine fine)
#   "thread"  ThreadSanitizer (cannot be combined with ASan)

function(annex_enable_sanitizers kind)
  if(kind STREQUAL "")
    return()
  endif()
  if(MSVC)
    message(FATAL_ERROR "annex: ANNEX_SANITIZE is only supported with GCC and Clang")
  endif()

  if(kind STREQUAL "address")
    # -fno-sanitize-recover makes UBSan abort instead of printing and carrying
    # on, so a UB report fails the test rather than scrolling past.
    set(flags -fsanitize=address,undefined -fno-sanitize-recover=all)
  elseif(kind STREQUAL "thread")
    set(flags -fsanitize=thread)
  else()
    message(FATAL_ERROR "annex: unknown ANNEX_SANITIZE='${kind}' (expected '', 'address' or 'thread')")
  endif()

  # Frame pointers give readable stack traces in sanitizer reports.
  add_compile_options(${flags} -fno-omit-frame-pointer)
  add_link_options(${flags})
  message(STATUS "annex: sanitizer enabled: ${kind}")
endfunction()
