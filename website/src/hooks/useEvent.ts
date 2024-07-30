import { useCallback, useLayoutEffect, useRef } from "react";

/**
 * useEvent implementation based on RFC proposal.
 * See: https://github.com/reactjs/rfcs/blob/d85e257502a43c08d17e8ab58efa0880f7f007a5/text/0000-useevent.md
 *
 * useEvent provides a stable function reference without having to deal with stale closure issues
 *
 * @param handler
 */
// Function is explicitly desired here
// eslint-disable-next-line @typescript-eslint/ban-types
function useEvent<T extends Function>(handler: T): T {
  const handlerRef = useRef<T | null>(null);

  // In a real implementation, this would run before layout effects
  useLayoutEffect(() => {
    handlerRef.current = handler;
  });

  const callback = useCallback((...args: any) => {
    if (!handlerRef.current) {
      throw new Error("event called before initialization");
    }
    // In a real implementation, this would throw if called during render
    const fn = handlerRef.current;
    return fn(...args);
  }, []);
  return callback as unknown as T;
}

export default useEvent;
