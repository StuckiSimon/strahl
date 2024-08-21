export function getStandardDeviation(list: number[]) {
  if (list.length === 0) {
    return 0;
  }
  const n = list.length;
  const mean = getSampleMean(list);
  return Math.sqrt(
    list.map((x) => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n,
  );
}

export function getSampleMean(list: number[]) {
  if (list.length === 0) {
    return 0;
  }
  return list.reduce((a, b) => a + b) / list.length;
}

export function getConfidenceInterval(
  xBar: number,
  s: number,
  n: number,
  z: number,
) {
  const marginOfError = z * (s / Math.sqrt(n));

  const lowerBound = xBar - marginOfError;
  const upperBound = xBar + marginOfError;

  return { lowerBound, upperBound, marginOfError };
}
