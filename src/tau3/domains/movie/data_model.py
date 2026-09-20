"""Data models for the movie ticketing domain.

The movie ticketing domain mirrors the structure of the airline domain, using
movie-native vocabulary (see ``data/tau3/domains/movie/policy.md``):

    airline concept        movie concept
    ---------------------  ---------------------
    flight / flight_number showtime / show_id
    origin (IATA)          theater
    destination (IATA)     movie_name
    cabin class            seat_class (standard / premium / vip)
    passenger              attendee
    baggage                snack
    certificate            movie_voucher

The database (``db.json``) currently only contains ``showtimes``. ``users`` and
``reservations`` therefore default to empty dictionaries so the existing data
loads as-is; they can be populated later without any code changes.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from tau3.environment.db import DB
from tau3.utils.utils import DATA_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# NOTE: For parity with the other domains these constants normally live in
# ``utils.py``. They are defined here so that ``data_model.py`` and ``tools.py``
# are self-contained and runnable on their own.
MOVIE_DATA_DIR = DATA_DIR / "tau3" / "domains" / "movie"
MOVIE_DB_PATH = MOVIE_DATA_DIR / "db.json"
MOVIE_POLICY_PATH = MOVIE_DATA_DIR / "policy.md"

# ---------------------------------------------------------------------------
# Enumerations / simple aliases
# ---------------------------------------------------------------------------
SeatClass = Literal["standard", "premium", "vip"]
Insurance = Literal["yes", "no"]

MembershipLevel = Annotated[
    Literal["gold", "silver", "regular"], Field(description="Membership level")
]


# ---------------------------------------------------------------------------
# User sub-models
# ---------------------------------------------------------------------------
class Name(BaseModel):
    first_name: str = Field(description="The person's first name")
    last_name: str = Field(description="The person's last name")


class Address(BaseModel):
    address1: str = Field(description="Primary address line")
    address2: Optional[str] = Field(
        None, description="Secondary address line (optional)"
    )
    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    state: str = Field(description="State or province name")
    zip: str = Field(description="Postal code")


# ---------------------------------------------------------------------------
# Payment related models
# ---------------------------------------------------------------------------
class Payment(BaseModel):
    payment_id: str = Field(description="Unique identifier for the payment method")
    amount: int = Field(description="Payment amount in dollars")


class PaymentMethodBase(BaseModel):
    source: str = Field(description="Type of payment method")
    id: str = Field(description="Unique identifier for the payment method")


class CreditCard(PaymentMethodBase):
    source: Literal["credit_card"] = Field(
        description="Indicates this is a credit card payment method"
    )
    brand: str = Field(description="Credit card brand (e.g., visa, mastercard)")
    last_four: str = Field(description="Last four digits of the credit card")


class GiftCard(PaymentMethodBase):
    source: Literal["gift_card"] = Field(
        description="Indicates this is a gift card payment method"
    )
    amount: float = Field(description="Gift card value amount")
    id: str = Field(description="Unique identifier for the gift card")


class MovieVoucher(PaymentMethodBase):
    source: Literal["movie_voucher"] = Field(
        description="Indicates this is a movie voucher payment method"
    )
    amount: float = Field(description="Movie voucher value amount")


PaymentMethod = Union[CreditCard, GiftCard, MovieVoucher]


# ---------------------------------------------------------------------------
# Attendee
# ---------------------------------------------------------------------------
class Attendee(BaseModel):
    first_name: str = Field(description="Attendee's first name")
    last_name: str = Field(description="Attendee's last name")
    dob: str = Field(description="Date of birth in YYYY-MM-DD format")


# ---------------------------------------------------------------------------
# Seat pricing / availability
# ---------------------------------------------------------------------------
SeatPrices = Annotated[
    Dict[SeatClass, int], Field(description="Prices for different seat classes")
]
AvailableSeats = Annotated[
    Dict[SeatClass, int],
    Field(description="Available seats for different seat classes"),
]


# ---------------------------------------------------------------------------
# Showtime date-status models
# ---------------------------------------------------------------------------
class ShowtimeDateStatusAvailable(BaseModel):
    status: Literal["available"] = Field(
        description="Indicates the showtime is available for booking"
    )
    available_seats: AvailableSeats = Field(
        description="Available seats by seat class"
    )
    prices: SeatPrices = Field(description="Current prices by seat class")


class ShowtimeDateStatusOnTime(BaseModel):
    status: Literal["on time"] = Field(
        description="Indicates the showtime is on time (not yet started, not bookable)"
    )
    estimated_start_time: str = Field(
        description="Estimated start time in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-16T14:00:00"
    )
    estimated_end_time: str = Field(
        description="Estimated end time in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-16T16:45:00"
    )


class ShowtimeDateStatusDelayed(BaseModel):
    status: Literal["delayed"] = Field(
        description="Indicates the showtime is delayed (not yet started, not bookable)"
    )
    estimated_start_time: str = Field(
        description="Estimated start time in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-16T14:00:00"
    )
    estimated_end_time: str = Field(
        description="Estimated end time in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-16T16:45:00"
    )


class ShowtimeDateStatusPlaying(BaseModel):
    status: Literal["playing"] = Field(
        description="Indicates the movie has started but not finished (not bookable)"
    )
    actual_start_time: str = Field(
        description="Actual start time in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T14:00:00"
    )
    estimated_end_time: str = Field(
        description="Estimated end time in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T16:45:00"
    )


class ShowtimeDateStatusPlayed(BaseModel):
    status: Literal["played"] = Field(
        description="Indicates the movie has finished playing"
    )
    actual_start_time: str = Field(
        description="Actual start time in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-01T14:05:00"
    )
    actual_end_time: str = Field(
        description="Actual end time in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-01T16:50:00"
    )


class ShowtimeDateStatusCancelled(BaseModel):
    status: Literal["cancelled"] = Field(
        description="Indicates the showtime was cancelled"
    )


ShowtimeDateStatus = Union[
    ShowtimeDateStatusAvailable,
    ShowtimeDateStatusOnTime,
    ShowtimeDateStatusDelayed,
    ShowtimeDateStatusPlaying,
    ShowtimeDateStatusPlayed,
    ShowtimeDateStatusCancelled,
]


# ---------------------------------------------------------------------------
# Showtime models
# ---------------------------------------------------------------------------
class ShowtimeBase(BaseModel):
    show_id: str = Field(description="Unique showtime identifier, such as 'MOV001'")
    theater: str = Field(description="Name of the theater")
    movie_name: str = Field(description="Name of the movie")


class Showtime(ShowtimeBase):
    scheduled_start_time: str = Field(
        description="Scheduled local start time in the format HH:MM:SS, e.g 14:00:00"
    )
    scheduled_end_time: str = Field(
        description="Scheduled local end time in the format HH:MM:SS, e.g 16:45:00"
    )
    dates: Dict[str, ShowtimeDateStatus] = Field(
        description="Showtime status by date (YYYY-MM-DD)"
    )


class DirectShowtime(ShowtimeBase):
    """A single bookable showtime instance, as returned by search."""

    status: Literal["available"] = Field(
        description="Indicates the showtime is available for booking"
    )
    scheduled_start_time: str = Field(
        description="Scheduled local start time in the format HH:MM:SS, e.g 14:00:00"
    )
    scheduled_end_time: str = Field(
        description="Scheduled local end time in the format HH:MM:SS, e.g 16:45:00"
    )
    date: Optional[str] = Field(
        description="Showtime date in YYYY-MM-DD format", default=None
    )
    available_seats: AvailableSeats = Field(
        description="Available seats by seat class"
    )
    prices: SeatPrices = Field(description="Current prices by seat class")


class ReservationShowtime(ShowtimeBase):
    """A showtime as stored inside a reservation."""

    date: str = Field(description="Showtime date in YYYY-MM-DD format")
    price: int = Field(description="Ticket price per attendee in dollars")


class ShowtimeInfo(BaseModel):
    """A lightweight reference to a specific showtime instance."""

    show_id: str = Field(description="Showtime id, such as 'MOV001'")
    date: str = Field(
        description="The date for the showtime in the format 'YYYY-MM-DD', such as '2024-05-16'"
    )


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------
class User(BaseModel):
    user_id: str = Field(description="Unique identifier for the user")
    name: Name = Field(description="User's full name")
    address: Address = Field(description="User's address information")
    email: str = Field(description="User's email address")
    dob: str = Field(
        description="User's date of birth in the format YYYY-MM-DD, e.g 1990-04-05"
    )
    payment_methods: Dict[str, PaymentMethod] = Field(
        description="User's saved payment methods"
    )
    saved_attendees: List[Attendee] = Field(
        default_factory=list, description="User's saved attendee information"
    )
    membership: MembershipLevel = Field(description="User's membership level")
    reservations: List[str] = Field(
        default_factory=list, description="List of user's reservation IDs"
    )


# ---------------------------------------------------------------------------
# Reservation
# ---------------------------------------------------------------------------
class Reservation(BaseModel):
    reservation_id: str = Field(description="Unique identifier for the reservation")
    user_id: str = Field(description="ID of the user who made the reservation")
    theater: str = Field(description="Theater the reservation is for")
    movie_name: str = Field(description="Movie the reservation is for")
    seat_class: SeatClass = Field(description="Selected seat class")
    showtimes: List[ReservationShowtime] = Field(
        description="List of showtimes in the reservation"
    )
    attendees: List[Attendee] = Field(
        description="List of attendees on the reservation"
    )
    payment_history: List[Payment] = Field(
        description="History of payments for this reservation"
    )
    created_at: str = Field(
        description="Timestamp when reservation was created in the format YYYY-MM-DDTHH:MM:SS"
    )
    total_snacks: int = Field(description="Total number of snack combos in reservation")
    nonfree_snacks: int = Field(
        description="Number of paid (non-free) snack combos in reservation"
    )
    insurance: Insurance = Field(
        description="Whether ticket insurance was purchased"
    )
    status: Optional[Literal["cancelled"]] = Field(
        description="Status of the reservation", default=None
    )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
class MovieDB(DB):
    """Database of all showtimes, users, and reservations."""

    showtimes: Dict[str, Showtime] = Field(
        description="Dictionary of all showtimes indexed by show id"
    )
    users: Dict[str, User] = Field(
        default_factory=dict,
        description="Dictionary of all users indexed by user ID",
    )
    reservations: Dict[str, Reservation] = Field(
        default_factory=dict,
        description="Dictionary of all reservations indexed by reservation ID",
    )

    def get_statistics(self) -> dict[str, Any]:
        """Get the statistics of the database."""
        num_showtimes = len(self.showtimes)
        num_showtime_instances = sum(
            len(showtime.dates) for showtime in self.showtimes.values()
        )
        num_users = len(self.users)
        num_reservations = len(self.reservations)
        return {
            "num_showtimes": num_showtimes,
            "num_showtime_instances": num_showtime_instances,
            "num_users": num_users,
            "num_reservations": num_reservations,
        }


def get_db() -> MovieDB:
    return MovieDB.load(MOVIE_DB_PATH)


if __name__ == "__main__":
    db = get_db()
    print(db.get_statistics())
