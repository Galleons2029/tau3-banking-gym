"""Toolkit for the movie ticketing system.

Mirrors the airline toolkit, remapped to movie-ticketing vocabulary (see
``data/tau3/domains/movie/policy.md``). As stated in the policy, the API is
intentionally "dumb": it does not enforce most business rules (who may modify a
standard ticket, cancellation eligibility, attendee caps, snack allowances,
etc.). The agent is responsible for applying those rules before calling a tool.
"""

from copy import deepcopy
from typing import List, Optional

from loguru import logger

from tau3.domains.movie.data_model import (
    Attendee,
    DirectShowtime,
    MovieDB,
    MovieVoucher,
    Payment,
    Reservation,
    ReservationShowtime,
    SeatClass,
    Showtime,
    ShowtimeDateStatus,
    ShowtimeDateStatusAvailable,
    ShowtimeInfo,
    Insurance,
    User,
)
from tau3.environment.toolkit import ToolKitBase, ToolType, is_tool

# Pricing constants (see policy.md).
SNACK_COMBO_PRICE = 10  # dollars per extra (non-free) snack combo
INSURANCE_PRICE_PER_ATTENDEE = 5  # dollars per attendee
VOUCHER_AMOUNT_PER_ATTENDEE_CANCELLED = 20  # compensation for a cancelled show
VOUCHER_AMOUNT_PER_ATTENDEE_DELAYED = 10  # compensation for a delayed show

# All theaters known to the system (kept in sync with the data generator).
THEATERS = [
    "Downtown AMC",
    "Uptown Regal",
    "Cinemark Central",
    "Grand Cineplex",
    "Riverside IMAX",
    "Midtown Odeon",
    "Harbor Cinemas",
    "Sunset Drive-In",
    "Metro Screen 8",
    "Palace Theater",
]


class MovieTools(ToolKitBase):
    """All the tools for the movie ticketing domain."""

    db: MovieDB

    def __init__(self, db: MovieDB) -> None:
        super().__init__(db)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_user(self, user_id: str) -> User:
        """Get user from database."""
        if user_id not in self.db.users:
            raise ValueError(f"User {user_id} not found")
        return self.db.users[user_id]

    def _get_reservation(self, reservation_id: str) -> Reservation:
        """Get reservation from database."""
        if reservation_id not in self.db.reservations:
            raise ValueError(f"Reservation {reservation_id} not found")
        return self.db.reservations[reservation_id]

    def _get_showtime(self, show_id: str) -> Showtime:
        """Get showtime from database."""
        if show_id not in self.db.showtimes:
            raise ValueError(f"Showtime {show_id} not found")
        return self.db.showtimes[show_id]

    def _get_showtime_instance(self, show_id: str, date: str) -> ShowtimeDateStatus:
        """Get showtime instance (a specific date) from database."""
        showtime = self._get_showtime(show_id)
        if date not in showtime.dates:
            raise ValueError(f"Showtime {show_id} not found on date {date}")
        return showtime.dates[date]

    def _get_new_reservation_id(self) -> str:
        """Get a new reservation id.

        Assumes each task makes at most 3 reservations.

        Returns:
            A new reservation id.

        Raises:
            ValueError: If too many reservations are made.
        """
        for reservation_id in ["TICKET", "TICKEU", "TICKEV"]:
            if reservation_id not in self.db.reservations:
                return reservation_id
        raise ValueError("Too many reservations")

    def _get_new_payment_id(self) -> list[int]:
        """Get candidate ids for new payment methods.

        Assumes each task creates at most 3 payment methods.

        Returns:
            A list of candidate numeric ids.
        """
        return [3221322, 3221323, 3221324]

    def _get_datetime(self) -> str:
        """Get the current datetime.

        Kept consistent with the reference date used to generate the data
        (2024-05-16) so that ``created_at`` timestamps line up with bookable
        showtimes.
        """
        return "2024-05-16T14:00:00"

    def _search_showtime(
        self,
        date: str,
        theater: Optional[str] = None,
        movie_name: Optional[str] = None,
    ) -> list[DirectShowtime]:
        """Search for available showtimes on a given date.

        Args:
            date: The date in the format 'YYYY-MM-DD', such as '2024-05-16'.
            theater: The theater name to filter by, such as 'Downtown AMC'.
            movie_name: The movie name to filter by, such as 'Dune: Part Two'.
        """
        results = []
        for showtime in self.db.showtimes.values():
            check = (
                (theater is None or showtime.theater == theater)
                and (movie_name is None or showtime.movie_name == movie_name)
                and (date in showtime.dates)
                and (showtime.dates[date].status == "available")
            )
            if check:
                date_data = showtime.dates[date]
                results.append(
                    DirectShowtime(
                        show_id=showtime.show_id,
                        theater=showtime.theater,
                        movie_name=showtime.movie_name,
                        status="available",
                        scheduled_start_time=showtime.scheduled_start_time,
                        scheduled_end_time=showtime.scheduled_end_time,
                        date=date,
                        available_seats=date_data.available_seats,
                        prices=date_data.prices,
                    )
                )
        return results

    def _payment_for_update(
        self, user: User, payment_id: str, total_price: int
    ) -> Optional[Payment]:
        """Process a payment (or refund) for a reservation update.

        Args:
            user: The user to process payment for.
            payment_id: The payment method id to process.
            total_price: The amount to charge (positive) or refund (negative).

        Raises:
            ValueError: If the payment method is not found.
            ValueError: If a movie voucher is used to update a reservation.
            ValueError: If the gift card balance is not enough.
        """
        if payment_id not in user.payment_methods:
            raise ValueError("Payment method not found")
        payment_method = user.payment_methods[payment_id]
        if payment_method.source == "movie_voucher":
            raise ValueError("Movie voucher cannot be used to update reservation")
        elif (
            payment_method.source == "gift_card" and payment_method.amount < total_price
        ):
            raise ValueError("Gift card balance is not enough")

        # Deduct payment from gift card balance
        if payment_method.source == "gift_card":
            payment_method.amount -= total_price

        payment = None
        if total_price != 0:
            payment = Payment(payment_id=payment_id, amount=total_price)
        return payment

    # ------------------------------------------------------------------
    # Read tools
    # ------------------------------------------------------------------
    @is_tool(ToolType.READ)
    def get_user_details(self, user_id: str) -> User:
        """
        Get the details of a user, including their reservations.

        Args:
            user_id: The user ID, such as 'sara_doe_496'.

        Returns:
            The user details.

        Raises:
            ValueError: If the user is not found.
        """
        return self._get_user(user_id)

    @is_tool(ToolType.READ)
    def get_reservation_details(self, reservation_id: str) -> Reservation:
        """
        Get the details of a reservation.

        Args:
            reservation_id: The reservation ID, such as 'TICKET'.

        Returns:
            The reservation details.

        Raises:
            ValueError: If the reservation is not found.
        """
        return self._get_reservation(reservation_id)

    @is_tool(ToolType.READ)
    def list_all_theaters(self) -> list[str]:
        """Returns a list of all available theaters.

        Returns:
            A list of theater names.
        """
        return list(THEATERS)

    @is_tool(ToolType.READ)
    def search_showtime(
        self, theater: str, movie_name: str, date: str
    ) -> list[DirectShowtime]:
        """
        Search for available showtimes of a movie at a theater on a specific date.

        Args:
            theater: The theater name, such as 'Downtown AMC'.
            movie_name: The movie name, such as 'Dune: Part Two'.
            date: The date in the format 'YYYY-MM-DD', such as '2024-05-16'.

        Returns:
            The available showtimes matching the theater, movie, and date.
        """
        return self._search_showtime(
            date=date, theater=theater, movie_name=movie_name
        )

    @is_tool(ToolType.READ)
    def get_showtime_status(self, show_id: str, date: str) -> str:
        """
        Get the status of a showtime on a specific date.

        Args:
            show_id: The showtime id, such as 'MOV001'.
            date: The date of the showtime, such as '2024-05-16'.

        Returns:
            The status of the showtime (e.g. 'available', 'playing', 'played', 'cancelled').

        Raises:
            ValueError: If the showtime is not found.
        """
        return self._get_showtime_instance(show_id, date).status

    # ------------------------------------------------------------------
    # Write tools
    # ------------------------------------------------------------------
    @is_tool(ToolType.WRITE)
    def book_reservation(
        self,
        user_id: str,
        theater: str,
        movie_name: str,
        seat_class: SeatClass,
        showtimes: List[ShowtimeInfo | dict],
        attendees: List[Attendee | dict],
        payment_methods: List[Payment | dict],
        total_snacks: int,
        nonfree_snacks: int,
        insurance: Insurance,
    ) -> Reservation:
        """
        Book a movie ticket reservation.

        Args:
            user_id: The ID of the user to book the reservation, such as 'sara_doe_496'.
            theater: The theater name, such as 'Downtown AMC'.
            movie_name: The movie name, such as 'Dune: Part Two'.
            seat_class: The seat class such as 'standard', 'premium', or 'vip'.
            showtimes: An array of objects, each identifying a showtime by show_id and date.
            attendees: An array of objects containing details about each attendee.
            payment_methods: An array of objects containing details about each payment method.
            total_snacks: The total number of snack combos to book.
            nonfree_snacks: The number of paid (non-free) snack combos to book.
            insurance: Whether the reservation includes ticket insurance ('yes' or 'no').

        Returns:
            The created reservation.

        Raises:
            ValueError: If the user is not found.
            ValueError: If a showtime is not available on the requested date.
            ValueError: If there are not enough available seats.
            ValueError: If a payment method is not found or has insufficient balance.
            ValueError: If the total payment does not match the total price.
        """
        if all(isinstance(showtime, dict) for showtime in showtimes):
            showtimes = [ShowtimeInfo(**showtime) for showtime in showtimes]
        if all(isinstance(attendee, dict) for attendee in attendees):
            attendees = [Attendee(**attendee) for attendee in attendees]
        if all(isinstance(payment_method, dict) for payment_method in payment_methods):
            payment_methods = [
                Payment(**payment_method) for payment_method in payment_methods
            ]
        user = self._get_user(user_id)
        reservation_id = self._get_new_reservation_id()

        reservation = Reservation(
            reservation_id=reservation_id,
            user_id=user_id,
            theater=theater,
            movie_name=movie_name,
            seat_class=seat_class,
            showtimes=[],
            attendees=deepcopy(attendees),
            payment_history=deepcopy(payment_methods),
            created_at=self._get_datetime(),
            total_snacks=total_snacks,
            nonfree_snacks=nonfree_snacks,
            insurance=insurance,
        )

        # Update showtimes and calculate price
        total_price = 0
        all_showtime_date_data: list[ShowtimeDateStatusAvailable] = []

        for showtime_info in showtimes:
            show_id = showtime_info.show_id
            showtime = self._get_showtime(show_id)
            showtime_date_data = self._get_showtime_instance(
                show_id=show_id, date=showtime_info.date
            )
            # Check showtime availability
            if not isinstance(showtime_date_data, ShowtimeDateStatusAvailable):
                raise ValueError(
                    f"Showtime {show_id} not available on date {showtime_info.date}"
                )
            # Check seat availability
            if showtime_date_data.available_seats[seat_class] < len(attendees):
                raise ValueError(f"Not enough seats on showtime {show_id}")
            # Calculate price
            price = showtime_date_data.prices[seat_class]
            reservation.showtimes.append(
                ReservationShowtime(
                    show_id=show_id,
                    theater=showtime.theater,
                    movie_name=showtime.movie_name,
                    date=showtime_info.date,
                    price=price,
                )
            )
            all_showtime_date_data.append(showtime_date_data)
            total_price += price * len(attendees)

        # Add insurance fee
        if insurance == "yes":
            total_price += INSURANCE_PRICE_PER_ATTENDEE * len(attendees)

        # Add snack fee
        total_price += SNACK_COMBO_PRICE * nonfree_snacks

        # Validate payment methods exist and have sufficient balance
        for payment_method in payment_methods:
            payment_id = payment_method.payment_id
            amount = payment_method.amount
            if payment_id not in user.payment_methods:
                raise ValueError(f"Payment method {payment_id} not found")
            user_payment_method = user.payment_methods[payment_id]
            if user_payment_method.source in {"gift_card", "movie_voucher"}:
                if user_payment_method.amount < amount:
                    raise ValueError(
                        f"Not enough balance in payment method {payment_id}"
                    )

        total_payment = sum(payment.amount for payment in payment_methods)
        if total_payment != total_price:
            raise ValueError(
                f"Payment amount does not add up, total price is {total_price}, but paid {total_payment}"
            )

        # If checks pass, deduct payment
        for payment_method in payment_methods:
            payment_id = payment_method.payment_id
            amount = payment_method.amount
            user_payment_method = user.payment_methods[payment_id]
            if user_payment_method.source == "gift_card":
                user_payment_method.amount -= amount
            elif user_payment_method.source == "movie_voucher":
                # The remaining amount of a movie voucher is not refundable:
                # using it consumes the whole voucher.
                user.payment_methods.pop(payment_id)

        # Update DB
        for showtime_date_data in all_showtime_date_data:
            showtime_date_data.available_seats[seat_class] -= len(attendees)
        self.db.reservations[reservation_id] = reservation
        self.db.users[user_id].reservations.append(reservation_id)
        return reservation

    @is_tool(ToolType.WRITE)
    def update_reservation_showtimes(
        self,
        reservation_id: str,
        seat_class: SeatClass,
        showtimes: List[ShowtimeInfo | dict],
        payment_id: str,
    ) -> Reservation:
        """
        Update the showtimes and/or seat class of a reservation.

        Args:
            reservation_id: The reservation ID, such as 'TICKET'.
            seat_class: The seat class of the reservation ('standard', 'premium', or 'vip').
            showtimes: An array of objects identifying every showtime in the ENTIRE new
                reservation. Even a showtime that is not being changed must still be included.
            payment_id: The payment id stored in the user profile, such as 'credit_card_7815826'
                or 'gift_card_7815826'.

        Returns:
            The updated reservation.

        Raises:
            ValueError: If the reservation is not found.
            ValueError: If the user is not found.
            ValueError: If a showtime is not available on the requested date.
            ValueError: If there are not enough available seats.
            ValueError: If the payment method is not found.
            ValueError: If a movie voucher is used to update the reservation.
            ValueError: If the gift card balance is not enough.
        """
        if all(isinstance(showtime, dict) for showtime in showtimes):
            showtimes = [ShowtimeInfo(**showtime) for showtime in showtimes]
        reservation = self._get_reservation(reservation_id)
        user = self._get_user(reservation.user_id)

        # Update showtimes and calculate price
        total_price = 0
        reservation_showtimes = []
        for showtime_info in showtimes:
            # Keep an existing, unchanged showtime (same show_id, date, and seat class)
            matching_reservation_showtime = next(
                (
                    reservation_showtime
                    for reservation_showtime in reservation.showtimes
                    if reservation_showtime.show_id == showtime_info.show_id
                    and reservation_showtime.date == showtime_info.date
                    and seat_class == reservation.seat_class
                ),
                None,
            )
            if matching_reservation_showtime:
                total_price += matching_reservation_showtime.price * len(
                    reservation.attendees
                )
                reservation_showtimes.append(matching_reservation_showtime)
                continue

            # Otherwise this is a new showtime instance
            showtime = self._get_showtime(showtime_info.show_id)
            showtime_date_data = self._get_showtime_instance(
                show_id=showtime_info.show_id,
                date=showtime_info.date,
            )
            if not isinstance(showtime_date_data, ShowtimeDateStatusAvailable):
                raise ValueError(
                    f"Showtime {showtime_info.show_id} not available on date {showtime_info.date}"
                )
            if showtime_date_data.available_seats[seat_class] < len(
                reservation.attendees
            ):
                raise ValueError(
                    f"Not enough seats on showtime {showtime_info.show_id}"
                )

            reservation_showtime = ReservationShowtime(
                show_id=showtime_info.show_id,
                theater=showtime.theater,
                movie_name=showtime.movie_name,
                date=showtime_info.date,
                price=showtime_date_data.prices[seat_class],
            )
            total_price += reservation_showtime.price * len(reservation.attendees)
            reservation_showtimes.append(reservation_showtime)

        # Deduct amount already paid for the reservation
        total_price -= sum(
            showtime.price for showtime in reservation.showtimes
        ) * len(reservation.attendees)

        # Create payment (or refund) for the difference
        payment = self._payment_for_update(user, payment_id, total_price)
        if payment is not None:
            reservation.payment_history.append(payment)

        # Update reservation
        reservation.showtimes = reservation_showtimes
        reservation.seat_class = seat_class

        return reservation

    @is_tool(ToolType.WRITE)
    def update_reservation_snacks(
        self,
        reservation_id: str,
        total_snacks: int,
        nonfree_snacks: int,
        payment_id: str,
    ) -> Reservation:
        """
        Update the snack information of a reservation. Snacks can only be added, not removed.

        Args:
            reservation_id: The reservation ID, such as 'TICKET'.
            total_snacks: The updated total number of snack combos in the reservation.
            nonfree_snacks: The updated number of paid (non-free) snack combos in the reservation.
            payment_id: The payment id stored in the user profile, such as 'credit_card_7815826'
                or 'gift_card_7815826'.

        Returns:
            The updated reservation.

        Raises:
            ValueError: If the reservation is not found.
            ValueError: If the user is not found.
            ValueError: If the payment method is not found.
            ValueError: If a movie voucher is used to update the reservation.
            ValueError: If the gift card balance is not enough.
        """
        reservation = self._get_reservation(reservation_id)
        user = self._get_user(reservation.user_id)

        # Charge only for newly added paid snacks
        total_price = SNACK_COMBO_PRICE * max(
            0, nonfree_snacks - reservation.nonfree_snacks
        )

        payment = self._payment_for_update(user, payment_id, total_price)
        if payment is not None:
            reservation.payment_history.append(payment)

        reservation.total_snacks = total_snacks
        reservation.nonfree_snacks = nonfree_snacks

        return reservation

    @is_tool(ToolType.WRITE)
    def update_reservation_attendees(
        self, reservation_id: str, attendees: List[Attendee | dict]
    ) -> Reservation:
        """
        Update the attendee information of a reservation.

        The number of attendees cannot be changed, only their names/details.

        Args:
            reservation_id: The reservation ID, such as 'TICKET'.
            attendees: An array of objects containing details about each attendee.

        Returns:
            The updated reservation.

        Raises:
            ValueError: If the reservation is not found.
            ValueError: If the number of attendees does not match.
        """
        if all(isinstance(attendee, dict) for attendee in attendees):
            attendees = [Attendee(**attendee) for attendee in attendees]
        reservation = self._get_reservation(reservation_id)
        if len(attendees) != len(reservation.attendees):
            raise ValueError("Number of attendees does not match")
        reservation.attendees = deepcopy(attendees)
        return reservation

    @is_tool(ToolType.WRITE)
    def cancel_reservation(self, reservation_id: str) -> Reservation:
        """
        Cancel the whole reservation and refund the payments to the original methods.

        Args:
            reservation_id: The reservation ID, such as 'TICKET'.

        Returns:
            The updated (cancelled) reservation.

        Raises:
            ValueError: If the reservation is not found.
        """
        reservation = self._get_reservation(reservation_id)
        logger.debug(reservation.model_dump_json(indent=4))
        # Reverse the payments
        refunds = []
        for payment in reservation.payment_history:
            refunds.append(
                Payment(payment_id=payment.payment_id, amount=-payment.amount)
            )
        reservation.payment_history.extend(refunds)
        reservation.status = "cancelled"
        logger.debug(self._get_reservation(reservation_id).model_dump_json(indent=4))
        # Release seats
        logger.warning("Seats release not implemented for cancellation!!!")
        return reservation

    @is_tool(ToolType.WRITE)
    def send_movie_voucher(self, user_id: str, amount: int) -> str:
        """
        Send a movie voucher to a user as compensation. Be careful!

        Args:
            user_id: The ID of the user to send the voucher to, such as 'sara_doe_496'.
            amount: The dollar amount of the movie voucher to send.

        Returns:
            A message indicating the voucher was sent.

        Raises:
            ValueError: If the user is not found.
            ValueError: If too many vouchers are sent.
        """
        user = self._get_user(user_id)

        # Add a voucher, assume at most 3 per task
        for payment_id in [f"movie_voucher_{id}" for id in self._get_new_payment_id()]:
            if payment_id not in user.payment_methods:
                new_payment = MovieVoucher(
                    id=payment_id,
                    amount=amount,
                    source="movie_voucher",
                )
                user.payment_methods[payment_id] = new_payment
                return (
                    f"Movie voucher {payment_id} added to user {user_id} "
                    f"with amount {amount}."
                )
        raise ValueError("Too many vouchers")

    # ------------------------------------------------------------------
    # Generic tools
    # ------------------------------------------------------------------
    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """
        Calculate the result of a mathematical expression.

        Args:
            expression: The mathematical expression to calculate, such as '2 + 2'. The
                expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.

        Returns:
            The result of the mathematical expression.

        Raises:
            ValueError: If the expression is invalid.
        """
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent (theater manager), with a summary of the user's issue.
        Only transfer if
         -  the user explicitly asks for a human agent
         -  given the policy and the available tools, you cannot solve the user's issue.

        Args:
            summary: A summary of the user's issue.

        Returns:
            A message indicating the user has been transferred to a human agent.
        """
        return "Transfer successful"


if __name__ == "__main__":
    from tau3.domains.movie.data_model import MOVIE_DB_PATH

    movie = MovieTools(MovieDB.load(MOVIE_DB_PATH))
    print(movie.get_statistics())
