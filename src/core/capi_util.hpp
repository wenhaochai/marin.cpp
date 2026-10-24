// Helpers for the C ABI (capi.h): every nm_* entry point runs its body through capi_call, which turns exceptions into
// the status codes of capi.h (nm::DeviceError -> NM_ERR_DEVICE, nm::Fatal -> NM_ERR_INVALID, anything else ->
// NM_ERR_INTERNAL) and the thread-local message nm_last_error() returns.
#pragma once

#include <exception>
#include <string>

#include "marincpp/capi.h"
#include "marincpp/core.hpp"

namespace nm {

void capi_set_error(const std::string& msg);
void capi_clear_error();

template <class F>
int capi_call(F&& body) {
  try {
    capi_clear_error();
    body();
    return NM_OK;
  } catch (const DeviceError& e) {
    capi_set_error(e.what());
    return NM_ERR_DEVICE;
  } catch (const Fatal& e) {
    capi_set_error(e.what());
    return NM_ERR_INVALID;
  } catch (const std::exception& e) {
    capi_set_error(e.what());
  } catch (...) {
    capi_set_error("unknown exception");
  }
  return NM_ERR_INTERNAL;
}

}  // namespace nm
