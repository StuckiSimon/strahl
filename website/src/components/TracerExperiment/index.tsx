import React from "react";
import { Pane } from "tweakpane";
import { convertHexToRGB, OpenPBRMaterial } from "strahl";
import styles from "./styles.module.css";
import clsx from "clsx";
import usePathTracer from "@site/src/hooks/usePathTracer";

export default function TracerExperiment(): JSX.Element {
  const defaultColor = "#f20089";
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
    targetSamples: 500,
    clearColor: convertHexToRGB("#1B1B1D"),
    viewProjectionConfiguration: {
      matrixWorldContent: [
        -0.45178184301411944, 4.163336342344336e-17, 0.8921284472108064, 0,
        0.18290622579667423, 0.9787573022265018, 0.09262535237781978, 0,
        -0.8731772322315672, 0.20502229961225985, -0.44218477786341664, 0,
        -3.67881274400709, 0.6362064645963488, -1.879628578827991, 1,
      ],
      fov: 38.6701655,
      cameraTargetDistance: 4,
    },
    maxRayDepth: 3,
    environmentLightConfiguration: {
      sky: {
        power: 0.5,
        color: [0.8, 0.8, 1.0],
      },
      sun: {
        power: 1.0,
        angularSize: 35,
        latitude: 40,
        longitude: 160,
        color: [1.0, 1.0, 0.9],
      },
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
