import { isNil } from "../util/is-nil";

type ReportEntry = {
  bvhBuildTime: number;
  fullRenderLoopTime: number;
  allRenderTime: number;
  renderTimes: number[];
};

type ReportStructure = {
  max?: ReportEntry[];
  mid?: ReportEntry[];
  min?: ReportEntry[];
};

function getStandardDeviation(list: number[]) {
  if (list.length === 0) {
    return 0;
  }
  const n = list.length;
  const mean = list.reduce((a, b) => a + b) / n;
  return Math.sqrt(
    list.map((x) => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n,
  );
}

function getSampleMean(list: number[]) {
  if (list.length === 0) {
    return 0;
  }
  return list.reduce((a, b) => a + b) / list.length;
}

function confidenceInterval(xBar: number, s: number, n: number, z: number) {
  const marginOfError = z * (s / Math.sqrt(n));

  const lowerBound = xBar - marginOfError;
  const upperBound = xBar + marginOfError;

  return { lowerBound, upperBound, marginOfError };
}

function calculateConfidenceIntervalForSamples(items: number[]) {
  const xBar = getSampleMean(items);
  const s = getStandardDeviation(items);
  const n = items.length;
  const z = 1.96; // 95% confidence interval

  return confidenceInterval(xBar, s, n, z);
}

export default function getStatsForReportStructure(
  reportStructure: ReportStructure,
) {
  const knownKeys = ["max", "mid", "min"] as const;
  const MAX_KEY = 0;
  const MID_KEY = 1;
  const MIN_KEY = 2;
  const activeStructures = knownKeys.filter((key) => reportStructure[key]);

  function getReportData(key: (typeof knownKeys)[number]): ReportEntry[] {
    return reportStructure[key] ?? [];
  }

  const orderedAverageBvhBuildTimes = activeStructures.map((key) =>
    getSampleMean(getReportData(key).map((entry) => entry.bvhBuildTime)),
  );

  const orderedMarginOfErrorBvhBuildTimes = activeStructures.map(
    (key) =>
      calculateConfidenceIntervalForSamples(
        getReportData(key).map((entry) => entry.bvhBuildTime),
      ).marginOfError,
  );

  const orderedAverageAllRenderTimes = activeStructures.map((key) =>
    getSampleMean(getReportData(key).map((entry) => entry.allRenderTime)),
  );

  const orderedMarginOfErrorAllRenderTimes = activeStructures.map(
    (key) =>
      calculateConfidenceIntervalForSamples(
        getReportData(key).map((entry) => entry.allRenderTime),
      ).marginOfError,
  );

  const primaryPot = activeStructures[0];
  const renderTimePots = getReportData(primaryPot)
    .map((item) => item.renderTimes)
    .reduce<Record<string, number[]>>((agg, times) => {
      times.forEach((entry, i) => {
        if (isNil(agg[i])) {
          agg[i] = [];
        }
        agg[i].push(entry);
      });
      return agg;
    }, {});

  return {
    max: {
      averageBvhBuildTime: orderedAverageBvhBuildTimes[MAX_KEY]?.toFixed(2),
      marginOfErrorBvhBuildTime:
        orderedMarginOfErrorBvhBuildTimes[MAX_KEY]?.toFixed(2),
      averageAllRenderTime: orderedAverageAllRenderTimes[MAX_KEY]?.toFixed(2),
      marginOfErrorAllRenderTime:
        orderedMarginOfErrorAllRenderTimes[MAX_KEY]?.toFixed(2),
    },
    mid: {
      averageBvhBuildTime: orderedAverageBvhBuildTimes[MID_KEY]?.toFixed(2),
      marginOfErrorBvhBuildTime:
        orderedMarginOfErrorBvhBuildTimes[MID_KEY]?.toFixed(2),
      averageAllRenderTime: orderedAverageAllRenderTimes[MID_KEY]?.toFixed(2),
      marginOfErrorAllRenderTime:
        orderedMarginOfErrorAllRenderTimes[MID_KEY]?.toFixed(2),
    },
    min: {
      averageBvhBuildTime: orderedAverageBvhBuildTimes[MIN_KEY]?.toFixed(2),
      marginOfErrorBvhBuildTime:
        orderedMarginOfErrorBvhBuildTimes[MIN_KEY]?.toFixed(2),
      averageAllRenderTime: orderedAverageAllRenderTimes[MIN_KEY]?.toFixed(2),
      marginOfErrorAllRenderTime:
        orderedMarginOfErrorAllRenderTimes[MIN_KEY]?.toFixed(2),
    },
    renderTimeAnalysis: {
      ...Object.fromEntries(
        Object.entries(renderTimePots).map(([key, times]) => [
          key,
          {
            mean: getSampleMean(times).toFixed(2),
            marginOfError:
              calculateConfidenceIntervalForSamples(
                times,
              ).marginOfError.toFixed(2),
          },
        ]),
      ),
    },
  };
}
