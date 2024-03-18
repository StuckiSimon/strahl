export function logGroup(label: string) {
  const start = window.performance.now();

  return {
    end() {
      const end = window.performance.now();
      console.log(`${label}: ${end - start}ms`);
    },
  };
}
