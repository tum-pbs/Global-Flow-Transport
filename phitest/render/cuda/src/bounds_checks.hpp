
#pragma once

#ifndef _INCLUDE_BOUNDSCHECKS
#define _INCLUDE_BOUNDSCHECKS

//for bounds checking
#define CHECK_BOUNDS_SV3S(l, c1, v, c2, u) l c1 v.x && v.x c2 u && l c1 v.y && v.y c2 u && l c1 v.z && v.z c2 u
#define CHECK_BOUNDS_SV3V3(l, c1, v, c2, u) l c1 v.x && v.x c2 u.x && l c1 v.y && v.y c2 u.y && l c1 v.z && v.z c2 u.z
#define CHECK_BOUNDS_V3V3V3(l, c1, v, c2, u) l.x c1 v.x && v.x c2 u.x && l.x c1 v.y && v.y c2 u.y && l.x c1 v.z && v.z c2 u.z
#define CHECK_BOUND_SV3(v1, c, v2) v1 c v2.x && v1 c v2.y && v1 c v2.z
#define CHECK_BOUND_V3S(v1, c, v2) v1.x c v2 && v1.y c v2 && v1.z c v2
#define CHECK_BOUND_V3V3(v1, c, v2) v1.x c v2.x && v1.y c v2.y && v1 c v2.z

#endif //_INCLUDE_BOUNDSCHECKS