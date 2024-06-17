export function logGroup(label: string) {
  const start = window.performance.now();

  return {
    end() {
      const end = window.performance.now();
      const delta = end - start;
      console.log(`${label}: ${delta}ms`);
      return delta;
    },
  };
}
