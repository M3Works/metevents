from metevents.events import StormEvents
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Setup dates to look for precip events
    start = pd.to_datetime('2023-01-01')
    end = pd.to_datetime('2023-06-01')

    # Use metevents for pull them (comes from metevents.events)
    storms = StormEvents.from_station('TUM', start, end, source='CDEC')

    # Populate storms events list using find()
    storms.find()

    # Useful attributes. Number of events detected
    print(f"Number of storms: {storms.N}")

    # Each event has properties are of interest, storm events are CumulativePeriods so
    # ...there is a total attribute. Base period definitions live in metevents.periods
    for event in storms.events:
        print(f"\t{event.start} - {event.stop} ({event.duration}): {event.total}")

    # The structure of this can make plotting them convenient.
    fig, ax = plt.subplots(1)
    cumulative = storms.data.cumsum()
    top = cumulative.max()

    # Loop over the events and fill in between where our storms are
    for event in storms.events:
        ax.fill_between([event.start, event.stop], top, color='blue', alpha=0.2)

    # Plot over the accumulated precip data
    ax.plot(cumulative, color='black', label='Cumulative Precip at TUM')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
    