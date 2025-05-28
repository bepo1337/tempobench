from enum import Enum

COLUMN_NA = "N/A"

PLAYER_ID = "player_id"
REASON = "reason"
VALIDITY_START = "validity_start"
VALIDITY_END = "validity_end"
FIRST_NAME = "first_name"
LAST_NAME = "last_name"
PSEUDONYM = "pseudonym"
HEIGHT = "height"
DATE_OF_BIRTH = "date_of_birth"
AGE = "age"
FOOT = "foot"
POSITION = "position"
CITIZENSHIP = "citizenship"
INJURY = "injury"
INJURY_CATEGORY = "injury_category"
MARKET_VALUE = "market_value"
MARKET_VALUE_CATEGORY = "market_value_category"
LAST_TRANSFER_FEE = "last_transfer_fee"
CLUB = "club"
CLUB_ID = "club_id"
SEASON_ID = "season_id"
LEAGUE = "league"
LEAGUE_ID = "league_id"
INTERNATIONAL_COMPETITION = "international_competition"
COACH = "coach"
COACH_ID = "coach_id"
LEAGUE_PLAYED_MATCHES = "league_played_matches"
LEAGUE_MINUTES_PLAYED = "league_minutes_played"
LEAGUE_GOALS = "league_goals"
INTERNATIONAL_PLAYED_MATCHES = "international_played_matches"
INTERNATIONAL_MINUTES_PLAYED = "international_minutes_played"
INTERNATIONAL_GOALS = "international_goals"
MISSED_MATCHES ="missed_matches"

class ColumnType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1
    CATEGORICAL = 2
    DATE = 3
    ORDINAL = 5

column_to_type = {
    PLAYER_ID: ColumnType.CATEGORICAL,
    REASON: ColumnType.CATEGORICAL,
    VALIDITY_START: ColumnType.DATE,
    VALIDITY_END: ColumnType.DATE,
    FIRST_NAME: ColumnType.CATEGORICAL,
    LAST_NAME: ColumnType.CATEGORICAL,
    PSEUDONYM: ColumnType.CATEGORICAL,
    HEIGHT: ColumnType.CONTINUOUS,
    DATE_OF_BIRTH: ColumnType.DATE,
    AGE: ColumnType.DISCRETE,
    FOOT: ColumnType.CATEGORICAL,
    POSITION: ColumnType.CATEGORICAL,
    CITIZENSHIP: ColumnType.CATEGORICAL,
    INJURY: ColumnType.CATEGORICAL,
    INJURY_CATEGORY: ColumnType.CATEGORICAL,
    MARKET_VALUE: ColumnType.CONTINUOUS,
    MARKET_VALUE_CATEGORY: ColumnType.ORDINAL,
    LAST_TRANSFER_FEE: ColumnType.CONTINUOUS,
    CLUB: ColumnType.CATEGORICAL,
    CLUB_ID: ColumnType.CATEGORICAL,
    SEASON_ID: ColumnType.ORDINAL,
    LEAGUE: ColumnType.CATEGORICAL,
    LEAGUE_ID: ColumnType.CATEGORICAL,
    INTERNATIONAL_COMPETITION: ColumnType.ORDINAL,
    COACH: ColumnType.CATEGORICAL,
    COACH_ID: ColumnType.CATEGORICAL,
    LEAGUE_PLAYED_MATCHES: ColumnType.DISCRETE,
    LEAGUE_MINUTES_PLAYED: ColumnType.DISCRETE,
    LEAGUE_GOALS: ColumnType.DISCRETE,
    INTERNATIONAL_PLAYED_MATCHES: ColumnType.DISCRETE,
    INTERNATIONAL_MINUTES_PLAYED: ColumnType.DISCRETE,
    INTERNATIONAL_GOALS: ColumnType.DISCRETE,
    MISSED_MATCHES: ColumnType.DISCRETE
}

columns_with_ids = [
   PLAYER_ID,
    COACH_ID,
    CLUB_ID,
    LEAGUE_ID
]

all_num_col_names = [key for key, value in column_to_type.items() if value in {ColumnType.DISCRETE, ColumnType.CONTINUOUS}]
all_cat_col_names = [key for key, value in column_to_type.items() if value in {ColumnType.CATEGORICAL, ColumnType.ORDINAL}]
all_cat_col_without_ids = [key for key, value in column_to_type.items()
                           if value in {ColumnType.CATEGORICAL, ColumnType.ORDINAL}
                            and key not in columns_with_ids]
all_cat_cols_without_ids_and_player_names = [
    col for col in all_cat_col_without_ids if col not in {FIRST_NAME, LAST_NAME, PSEUDONYM}
]
class Category(str, Enum):
    SANITY = "SANITY"
    DOMAIN = "DOMAIN"
    STATISTICAL = "STATISTICAL"
    TEMPORAL = "TEMPORAL"
    UTILITY = "UTILITY"


columns_with_dates = [
    VALIDITY_START,
    VALIDITY_END,
    DATE_OF_BIRTH
]

performance_data = [
    LEAGUE_PLAYED_MATCHES,
    LEAGUE_MINUTES_PLAYED,
    LEAGUE_GOALS,
    INTERNATIONAL_PLAYED_MATCHES,
    INTERNATIONAL_MINUTES_PLAYED,
    INTERNATIONAL_GOALS
]

all_positions = [
  "Centre-Forward",
  "Second Striker",
  "Right Winger",
  "Left Winger",
  "Goalkeeper",
  "Attacking Midfield",
  "Defensive Midfield",
  "Right-Back",
  "Central Midfield",
  "Centre-Back",
  "Right Midfield",
  "Left-Back",
  "Left Midfield"
]

def is_numerical(col_type) -> bool:
    if col_type == ColumnType.DISCRETE or col_type == ColumnType.CONTINUOUS:
        return True

    return False

def is_categorical_or_ordinal(col_type) -> bool:
    if col_type == ColumnType.CATEGORICAL or col_type == ColumnType.ORDINAL:
        return True

    return False