using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace VI.AOD.Segmentation
{
    public class PythonScriptCaller
    {
        //public DataReceivedEventHandler OnSegmentationOutputRecieved { get; set; }
        public List<string> Error;
        private int timeout = 10000;

        public IList<string> RunUsingNewProcessInitialization(string cmd, Action<string> logOutput, string args = "")
        {
            var programFilesFolder = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);

            List<string> result = new List<string>();
            this.Error = new List<string>();

            //Process process = new Process();
            //process.StartInfo.FileName = Path.Combine(programFilesFolder, "Python36\\python.exe");
            //process.StartInfo.Arguments = string.Format("\"{0}\"{1}", cmd, args);
            //process.StartInfo.UseShellExecute = false;
            //process.StartInfo.RedirectStandardOutput = true;
            //process.StartInfo.RedirectStandardError = true;
            //process.StartInfo.CreateNoWindow = true;
            ////process.StartInfo.ErrorDialog = true;

            //process.OutputDataReceived += new DataReceivedEventHandler((sender, e) =>
            //{
            //    result.Add(e.Data);
            //});

            //process.ErrorDataReceived += new DataReceivedEventHandler((sender, e) =>
            //{
            //    Error.Add(e.Data);
            //});

            ////if(OnSegmentationOutputRecieved != null) process.OutputDataReceived += OnSegmentationOutputRecieved;

            //process.Start();

            //process.BeginErrorReadLine();
            //process.BeginOutputReadLine();
            //process.CancelErrorRead();
            //process.CancelOutputRead();
            //process.BeginErrorReadLine();            
            //process.BeginOutputReadLine();

            //process.WaitForExit();
            //process.Close();

            //string err = process.StandardError.ReadToEnd();            

            using (Process process = new Process())
            {
                process.StartInfo.FileName = Path.Combine(programFilesFolder, "Python36\\python.exe");
                process.StartInfo.Arguments = string.Format("\"{0}\"{1}", cmd, args);
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.RedirectStandardOutput = true;
                process.StartInfo.RedirectStandardError = true;
                
                using (AutoResetEvent outputWaitHandle = new AutoResetEvent(false))
                using (AutoResetEvent errorWaitHandle = new AutoResetEvent(false))
                {
                    process.OutputDataReceived += (sender, e) => {
                        if (e.Data == null)
                        {
                            outputWaitHandle.Set();
                        }
                        else
                        {
                            result.Add(e.Data);
                            logOutput(e.Data);
                        }
                    };
                    process.ErrorDataReceived += (sender, e) =>
                    {
                        if (e.Data == null)
                        {
                            errorWaitHandle.Set();
                        }
                        else
                        {
                            Error.Add(e.Data);
                            logOutput(e.Data);
                        }
                    };

                    process.Start();

                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();

                    process.WaitForExit();
                    outputWaitHandle.WaitOne();
                    errorWaitHandle.WaitOne();
                    //using (StreamReader reader = process.StandardOutput)
                    //{
                    //    string res = reader.ReadToEnd();
                    //    // this prints 11
                    //    Console.Write(result);

                    //}
                    //{
                    //    // Process completed. Check process.ExitCode here.
                    //}
                    //else
                    //{
                    //    // Timed out.
                    //}
                }
            }

            return result;
        }
    }
}
