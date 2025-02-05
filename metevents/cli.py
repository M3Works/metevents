"""Console script for metloom."""
import argparse
import datetime
import pandas as pd
import sys 

# Local imports xs
import utilities
import events
def main():
    """Console script for metevents."""

    parser = argparse.ArgumentParser(
        description="Pull meteterological events from station(s)"
    )
    parser.add_argument(
        "--station", "-sta", required=True, type=str,
        help="station for to utilize for event" 
    )
    parser.add_argument('--source', '-src', default="CDEC", help="datasource for the station", choices=["NRCS", "mesowest", "CDEC"])

    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--wateryear', '-wy', default=utilities.current_water_year(), type=int)
    group.add_argument('--dates', '-d', default=None, nargs=2, type=lambda input: [datetime.datetime.strptime(s, '%Y-%m-%d') for s in input], metavar='YYYY-MM-DD')
    group.add_argument('--plot', '-p', default='output.png')
    group.add_argument('--csv', '-c', default='output.csv')

    subparsers = parser.add_subparsers(dest='subcommand')
    storm_parser = subparsers.add_parser(
        "storm", help="Find storm events"
    )
    storm_parser.add_argument('--instant-mass-to-start', '-ims', type=float, default=0.1, help="mass per time step to consider the beginning of a storm")
    storm_parser.add_argument('--min-storm-total', '-mst', type=float, default=0.5, help="total storm mass to be considered a complete storm")
    storm_parser.add_argument('--hours-to-stop', '-hts', type=float, default=24, help="minimum hours of mass less than instant threshold to end a storm")
    storm_parser.add_argument('--max-storm-hours', '-msh', type=float, default=336, help="maximum hours a storm can continue")

    args = parser.parse_args()
    if args.wateryear:
        args.dates = [datetime.datetime(args.wateryear-1, 10, 1), datetime.datetime(args.wateryear, 9, 30)]

    event = None 
    if args.subcommand == "storm":
        event = events.StormEvents.from_station(args.station, args.dates[0], args.dates[1], source=args.source)
        event.find(instant_mass_to_start=args.instant_mass_to_start, 
                    hours_to_stop=args.hours_to_stop, 
                    min_storm_total=args.min_storm_total, 
                    max_storm_hours=args.max_storm_hours,)

    if args.plot:        
        fig, ax = event.to_plot()
        fig.savefig(args.plot)
    if args.csv:
        df = event.to_dataframe()
        df.to_csv(args.csv, index=False)
        

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
