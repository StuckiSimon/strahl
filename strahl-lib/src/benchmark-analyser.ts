type ReportEntry = {
  bvhBuildTime: number;
  fullRenderLoopTime: number;
  allRenderTime: number;
};

type ReportStructure = {
  max: ReportEntry[];
  mid: ReportEntry[];
  min: ReportEntry[];
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
  const averageBvhBuildTimeMax = getSampleMean(
    reportStructure.max.map((entry) => entry.bvhBuildTime),
  );
  const averageBvhBuildTimeMid = getSampleMean(
    reportStructure.mid.map((entry) => entry.bvhBuildTime),
  );
  const averageBvhBuildTimeMin = getSampleMean(
    reportStructure.min.map((entry) => entry.bvhBuildTime),
  );
  const { marginOfError: marginOfErrorBvhBuildTimeMax } =
    calculateConfidenceIntervalForSamples(
      reportStructure.max.map((entry) => entry.bvhBuildTime),
    );

  const { marginOfError: marginOfErrorBvhBuildTimeMid } =
    calculateConfidenceIntervalForSamples(
      reportStructure.mid.map((entry) => entry.bvhBuildTime),
    );

  const { marginOfError: marginOfErrorBvhBuildTimeMin } =
    calculateConfidenceIntervalForSamples(
      reportStructure.min.map((entry) => entry.bvhBuildTime),
    );

  const averageAllRenderTimeMax = getSampleMean(
    reportStructure.max.map((entry) => entry.allRenderTime),
  );
  const averageAllRenderTimeMid = getSampleMean(
    reportStructure.mid.map((entry) => entry.allRenderTime),
  );
  const averageAllRenderTimeMin = getSampleMean(
    reportStructure.min.map((entry) => entry.allRenderTime),
  );
  const { marginOfError: marginOfErrorAllRenderTimeMax } =
    calculateConfidenceIntervalForSamples(
      reportStructure.max.map((entry) => entry.allRenderTime),
    );
  const { marginOfError: marginOfErrorAllRenderTimeMid } =
    calculateConfidenceIntervalForSamples(
      reportStructure.mid.map((entry) => entry.allRenderTime),
    );
  const { marginOfError: marginOfErrorAllRenderTimeMin } =
    calculateConfidenceIntervalForSamples(
      reportStructure.min.map((entry) => entry.allRenderTime),
    );

  return {
    max: {
      averageBvhBuildTime: averageBvhBuildTimeMax.toFixed(2),
      marginOfErrorBvhBuildTime: marginOfErrorBvhBuildTimeMax.toFixed(2),
      averageAllRenderTime: averageAllRenderTimeMax.toFixed(2),
      marginOfErrorAllRenderTime: marginOfErrorAllRenderTimeMax.toFixed(2),
    },
    mid: {
      averageBvhBuildTime: averageBvhBuildTimeMid.toFixed(2),
      marginOfErrorBvhBuildTime: marginOfErrorBvhBuildTimeMid.toFixed(2),
      averageAllRenderTime: averageAllRenderTimeMid.toFixed(2),
      marginOfErrorAllRenderTime: marginOfErrorAllRenderTimeMid.toFixed(2),
    },
    min: {
      averageBvhBuildTime: averageBvhBuildTimeMin.toFixed(2),
      marginOfErrorBvhBuildTime: marginOfErrorBvhBuildTimeMin.toFixed(2),
      averageAllRenderTime: averageAllRenderTimeMin.toFixed(2),
      marginOfErrorAllRenderTime: marginOfErrorAllRenderTimeMin.toFixed(2),
    },
  };
}
