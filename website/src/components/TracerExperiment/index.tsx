import React from "react";
import { Pane } from "tweakpane";
import { convertHexToRGB, OpenPBRMaterial } from "strahl";
import styles from "./styles.module.css";
import clsx from "clsx";
import usePathTracer from "@site/src/hooks/usePathTracer";

type PartialOpenPBRMaterialConfiguration = Partial<
  Record<keyof OpenPBRMaterial, unknown>
>;

type Props = {
  propertiesForConfiguration: (keyof OpenPBRMaterial)[];
  defaultMaterialProperties?: PartialOpenPBRMaterialConfiguration;
};

function convertNormalizedToHex(c: number) {
  var hex = Math.round(c * 255).toString(16);
  return hex.padStart(2, "0");
}

// todo: use exported type from strahl
function convertRGBToHex(c: ReturnType<typeof convertHexToRGB>) {
  return (
    "#" +
    convertNormalizedToHex(c[0]) +
    convertNormalizedToHex(c[1]) +
    convertNormalizedToHex(c[2])
  );
}

const defaultColor = "#f20089";

const defaultConfiguration: Partial<
  Record<
    keyof OpenPBRMaterial,
    {
      configKey: string;
      value: unknown;
      convertToPaneValue: (value: unknown) => unknown;
      convertToMaterialValue: (value: unknown) => unknown;
    }
  >
> = {
  oBaseColor: {
    configKey: "baseColor",
    value: convertHexToRGB(defaultColor),
    convertToPaneValue: (value) =>
      convertRGBToHex(value as ReturnType<typeof convertHexToRGB>),
    convertToMaterialValue: (value) => convertHexToRGB(value as string),
  },
};

export default function TracerExperiment({
  propertiesForConfiguration,
  defaultMaterialProperties,
}: Props): JSX.Element {
  const defaultMaterial = {
    ...Object.fromEntries(
      Object.entries(defaultConfiguration).map(([key, { value }]) => [
        key,
        value,
      ]),
    ),
    ...defaultMaterialProperties,
  };
  const buildMaterial = (overrides: PartialOpenPBRMaterialConfiguration) => {
    const material = new OpenPBRMaterial();
    for (const key of propertiesForConfiguration) {
      // todo: consider nicer way
      // @ts-ignore
      material[key] = overrides[key] ?? defaultMaterial[key];
    }
    return material;
  };

  const [materialMap, setMaterialMap] = React.useState({
    floor: (() => {
      let m = new OpenPBRMaterial();
      m.oSpecularWeight = 0.0;
      m.oBaseColor = [0.4, 0.4, 0.4];
      return m;
    })(),
    sphere: buildMaterial({}),
  });

  const [options] = React.useState<Parameters<typeof usePathTracer>[2]>({
    targetSamples: 1,
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
    const PARAMS = Object.fromEntries(
      Object.entries(defaultConfiguration).map(
        ([key, { convertToPaneValue, configKey }]) => [
          configKey,
          convertToPaneValue(defaultMaterial[key]),
        ],
      ),
    );

    const pane = new Pane({
      container: document.getElementById(`pane-${paneContainerId}`),
    });

    for (const property of propertiesForConfiguration) {
      pane.addBinding(PARAMS, defaultConfiguration[property].configKey);
    }

    paneRef.current = pane;

    function convertPaneToOpenPBRMaterial(): PartialOpenPBRMaterialConfiguration {
      return Object.fromEntries(
        Object.entries(defaultConfiguration).map(
          ([key, { convertToMaterialValue, configKey }]) => [
            key,
            convertToMaterialValue(PARAMS[configKey]),
          ],
        ),
      );
    }

    pane.on("change", (ev) => {
      setMaterialMap((prev) => {
        const newMaterialMap = { ...prev };

        newMaterialMap.sphere = buildMaterial(convertPaneToOpenPBRMaterial());
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
