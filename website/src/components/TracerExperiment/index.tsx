import React from "react";
import { Pane } from "tweakpane";
import { convertHexToRGB, OpenPBRMaterial } from "strahl";
import styles from "./styles.module.css";
import clsx from "clsx";
import usePathTracer from "@site/src/hooks/usePathTracer";

export default function TracerExperiment(): JSX.Element {
  const defaultColor = "#ff0055";
  const [materialMap, setMaterialMap] = React.useState({
    floor: (() => {
      let m = new OpenPBRMaterial();
      m.oSpecularWeight = 0.0;
      m.oBaseColor = [0.4, 0.4, 0.4];
      return m;
    })(),
    sphere: (() => {
      let m = new OpenPBRMaterial();
      m.oBaseColor = convertHexToRGB(defaultColor);
      return m;
    })(),
  });

  const [options] = React.useState<Parameters<typeof usePathTracer>[2]>({
    targetSamples: 200,
    viewProjectionConfiguration: {
      matrixWorldContent: [
        -0.48044932418825687, 2.081668171172168e-17, 0.877022489384991, 0,
        0.11121397919060463, 0.9919272066348073, 0.06092509803241454, 0,
        -0.8699424680515592, 0.12680858305993156, -0.47657075607163846, 0,
        -3.6120942140398014, 0.16869853590213923, -2.0157257065629848, 1,
      ],
      fov: 38.6701655,
      cameraTargetDistance: 4,
    },
  });

  const paneContainerId = React.useId();
  const paneRef = React.useRef<Pane | null>(null);
  React.useEffect(() => {
    const PARAMS = {
      baseColor: defaultColor,
    };

    const pane = new Pane({
      container: document.getElementById(`pane-${paneContainerId}`),
    });

    pane.addBinding(PARAMS, "baseColor");

    paneRef.current = pane;

    pane.on("change", (ev) => {
      setMaterialMap((prev) => {
        const newMaterialMap = { ...prev };
        const color = convertHexToRGB(PARAMS.baseColor);

        const material = new OpenPBRMaterial();
        material.oBaseColor = color;

        newMaterialMap.sphere = material;
        return newMaterialMap;
      });
    });

    return () => {
      pane.dispose();
    };
  }, []);

  const canvas = usePathTracer(
    "https://stuckisimon.github.io/strahl-sample-models/sphere-plane/demo1.glb",
    materialMap,
    options,
  );

  return (
    <div className={styles.container}>
      <div className="container">
        <div className="row">
          <div id={`pane-${paneContainerId}`} />
          <div className={clsx("col", styles.wrapper)}>{canvas}</div>
        </div>
      </div>
    </div>
  );
}
