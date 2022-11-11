#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum EvolveOutcome {
  ReachedEventsMax,
  ReachedTimeMax,
  ReachWallTimeMax,
  ReachedSizeMin,
  ReachedSizeMax,
  ReachedZeroRate,
} EvolveOutcome;

typedef struct Tile Tile;

typedef struct TileSet TileSet;

typedef uint64_t NumEvents;

typedef enum COption_NumEvents_Tag {
  Some_NumEvents,
  None_NumEvents,
} COption_NumEvents_Tag;

typedef struct COption_NumEvents {
  COption_NumEvents_Tag tag;
  union {
    struct {
      NumEvents some;
    };
  };
} COption_NumEvents;

typedef enum COption_f64_Tag {
  Some_f64,
  None_f64,
} COption_f64_Tag;

typedef struct COption_f64 {
  COption_f64_Tag tag;
  union {
    struct {
      double some;
    };
  };
} COption_f64;

typedef uint32_t NumTiles;

typedef enum COption_NumTiles_Tag {
  Some_NumTiles,
  None_NumTiles,
} COption_NumTiles_Tag;

typedef struct COption_NumTiles {
  COption_NumTiles_Tag tag;
  union {
    struct {
      NumTiles some;
    };
  };
} COption_NumTiles;

typedef struct EvolveBounds {
  /**
   * Stop if this number of events has taken place during this evolve call.
   */
  struct COption_NumEvents for_events;
  /**
   * Stop if this number of events has been reached in total for the state.
   */
  struct COption_NumEvents total_events;
  /**
   * Stop if this amount of (simulated) time has passed during this evolve call.
   */
  struct COption_f64 for_time;
  /**
   * Stop if this amount of (simulated) time has passed in total for the state.
   */
  struct COption_f64 total_time;
  /**
   * Stop if the number of tiles is equal to or less than this number.
   */
  struct COption_NumTiles size_min;
  /**
   * Stop if the number of tiles is equal to or greater than this number.
   */
  struct COption_NumTiles size_max;
  /**
   * Stop after this amount of (real) time has passed.
   */
  struct COption_f64 for_wall_time;
} EvolveBounds;

typedef struct CArrayView2_Tile {
  const Tile *data;
  uint64_t nrows;
  uint64_t ncols;
} CArrayView2_Tile;

struct TileSet *create_tileset_from_file(const char *s);

struct TileSet *create_tileset_from_json(const char *s);

void *create_simulation_from_tileset(const struct TileSet *t);

uintptr_t new_state(void *sim);

enum EvolveOutcome evolve_index(void *sim, uint64_t state, struct EvolveBounds bounds);

struct CArrayView2_Tile get_canvas_view(const void *sim, uint64_t state);
