using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VI.AOD.Segmentation.SegmentationService
{
    public class SegmentationService : ISegmentationService
    {
        private string mainScriptPath;
        private bool initialized = false;
        private string segmentationScriptName;
        private string graphDirectory;
        private string storageFolder;

        //Implement as a singleton

        public SegmentationService(string mainScriptName, string graphDirectory, string storageFolder)
        {
            this.storageFolder = storageFolder;
            this.mainScriptPath = Path.Combine(storageFolder, mainScriptName);
            this.graphDirectory = Path.Combine(storageFolder, graphDirectory);
            
        }

        public void Initialize()
        {
         
            //this.mainScriptPath = Resources.ResourceInitializer.InitializeProgramData();
        }

        public Dictionary<string, string[]> RunSegmentation(string samplesDirectory, string outputDirectory, string cudaVisibleDevices, int batchSize, Action<string> logOutput)
        {
            if (!this.initialized) this.Initialize();
            //string myPythonApp = @"D:\Docs\ISR_ inspekcja wizualna\multi_gpu_test.py";

            string args = this.createArgumentsString(this.graphDirectory, samplesDirectory, outputDirectory, cudaVisibleDevices, batchSize);

            //Infer infer = new Infer();

            PythonScriptCaller pythonScriptCaller = new PythonScriptCaller();

            var result = pythonScriptCaller.RunUsingNewProcessInitialization(this.mainScriptPath, logOutput, args);

            var parsedResult = this.parseResult(result);

            return parsedResult;
        }

        private Dictionary<string, string[]> parseResult(IList<string> result)
        {
            Dictionary<string, string[]> parsedResult = new Dictionary<string, string[]>();

            for (int i = 0; i < result.Count; i++)
            {
                if (result[i] != null)
                {
                    string line = result[i];
                    var elements = line.Split(' ');

                    if (elements[0] == "Output" && elements.Last() == "done.")
                    {
                        var previousLineElements = result[i - 1].Split(' ');
                        var nextLineElements = result[i + 1].Split(' ');

                        if (previousLineElements[0] == "File" && previousLineElements.Last() == "done." && nextLineElements[0] == "Output" && nextLineElements.Last() == "done.")
                        {
                            parsedResult.Add(previousLineElements[1].Replace('/', '\\'), new string[] { elements[4].Replace('/', '\\'), nextLineElements[4].Replace('/', '\\') });
                            i++;
                        }
                    }
                }
            }


            //File G:/ ISR / 2019_6_10_10_1_40 / avi / 0\camera0_2.avi done.
            //Output file with images: C:/ SummarySesionsMix / Result / ISR_30epochs_1120x1024 / 2019_6_10_10_1_40 / camera0_2_image.avi done.
            //Output file with labels: C:/ SummarySesionsMix / Result / ISR_30epochs_1120x1024 / 2019_6_10_10_1_40 / camera0_2_label.avi done.
            return parsedResult;
        }

        private string createArgumentsString(string graphDirectory, string samplesDirectory, string outputDirectory, string cudaVisibleDevices, int batchSize)
        {
            string args = $" --name={graphDirectory}";//.Replace('\\', '/')}";
            args += $" --input_dir={samplesDirectory}";//.Replace('\\', '/')}";
            args += $" --output_dir={outputDirectory}";//.Replace('\\', '/')}";
            args += $" --batch-size={batchSize}";
            args += $" --frames_buffer={batchSize}";
            args += $" --cuda_visible_devices={cudaVisibleDevices}";

            return args;
        }
    }
}
