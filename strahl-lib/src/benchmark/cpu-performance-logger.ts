export function logGroup() {
  const start = window.performance.now();

  return {
    end() {
      const end = window.performance.now();
      const delta = end - start;
      return delta;
    },
  };
}
