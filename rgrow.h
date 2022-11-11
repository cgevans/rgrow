#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

enum class EvolveOutcome {
  ReachedEventsMax,
  ReachedTimeMax,
  ReachWallTimeMax,
  ReachedSizeMin,
  ReachedSizeMax,
  ReachedZeroRate,
};

struct Tile;

struct TileSet;

using NumEvents = uint64_t;

template<typename T>
struct COption {
  enum class Tag {
    Some,
    None,
  };

  struct Some_Body {
    T _0;
  };

  Tag tag;
  union {
    Some_Body some;
  };
};

using NumTiles = uint32_t;

struct EvolveBounds {
  /// Stop if this number of events has taken place during this evolve call.
  COption<NumEvents> for_events;
  /// Stop if this number of events has been reached in total for the state.
  COption<NumEvents> total_events;
  /// Stop if this amount of (simulated) time has passed during this evolve call.
  COption<double> for_time;
  /// Stop if this amount of (simulated) time has passed in total for the state.
  COption<double> total_time;
  /// Stop if the number of tiles is equal to or less than this number.
  COption<NumTiles> size_min;
  /// Stop if the number of tiles is equal to or greater than this number.
  COption<NumTiles> size_max;
  /// Stop after this amount of (real) time has passed.
  COption<double> for_wall_time;
};

template<typename T>
struct CArrayView2 {
  const T *data;
  uintptr_t nrows;
  uintptr_t ncols;
};

extern "C" {

TileSet *create_tileset_from_json(const char *s);

void *create_simulation_from_tileset(const TileSet *t);

uintptr_t new_state(void *sim);

EvolveOutcome evolve(void *sim, uintptr_t state, EvolveBounds bounds);

CArrayView2<Tile> get_canvas_view(void *sim, uintptr_t state);

} // extern "C"
