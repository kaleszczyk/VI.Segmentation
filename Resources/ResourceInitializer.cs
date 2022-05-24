using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Resources;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Reflection;

namespace VI.AOD.Segmentation.Resources
{
    public static class ResourceInitializer
    {
        public static string InitializeProgramData()
        {
            string resourceStorageFolder = ApplicationGlobals.GetCommonAppDataPath();
                
            //https://www.codeproject.com/Tips/370232/Where-should-I-store-my-data

            ResourceManager rm = new ResourceManager(typeof(Resources));
            var resourceSet = rm.GetResourceSet(System.Globalization.CultureInfo.InvariantCulture, true, true);

            CheckDir(resourceStorageFolder);
            string mainScriptPath = string.Empty;           

            foreach (DictionaryEntry resource in resourceSet)
            {
                string extension = ".py";
                if((string)resource.Key == "LabelPainter")
                {
                    extension = ".exe";
                }                                
                else if ((string)resource.Key == "colours")
                {
                    extension = ".xml";
                }

                string filePath = Path.Combine(resourceStorageFolder, (string)resource.Key) + extension;

                if (!File.Exists(filePath))
                {
                    if (resource.Value is string)
                    {
                        File.WriteAllText(filePath, (string)resource.Value);
                    }
                    else
                    {
                        File.WriteAllBytes(filePath, (byte[])resource.Value);
                    }
                }

                if ((string)resource.Key == "infer")
                {
                    mainScriptPath = filePath;
                }
            }

            return mainScriptPath;
        }

        private static string getResourceStorageFolderPath()
        {
            string applicationName = System.Reflection.Assembly.GetExecutingAssembly().FullName;//
            return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData), applicationName);
        }

        private static string CheckDir(string dir)
        {
            if (!Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }
            return dir;
        }
    }

    public static class ApplicationGlobals
    {
        public static string GetUserAppDataPath()
        {
            string path = Environment.GetFolderPath(
                                        Environment.SpecialFolder.LocalApplicationData);
            path = System.IO.Path.Combine(path, GetCompanyName());
            path = System.IO.Path.Combine(path, GetProductName());

            return path;
        }

        public static string GetCommonAppDataPath()
        {
            string path = Environment.GetFolderPath(
                                        Environment.SpecialFolder.CommonApplicationData);
            path = System.IO.Path.Combine(path, GetCompanyName());
            path = System.IO.Path.Combine(path, GetProductName());

            return path;
        }

        public static string GetProductLabel()
        {
            string mode = string.Empty;

#if DEBUG
            mode = "D";
#else
                mode = "R"; 
#endif

            return string.Format("{0} ({1} {2})", GetProductName(), GetVersion().FileVersion, mode);
        }

        public static string GetProductName()
        {
            var app = Assembly.GetEntryAssembly();
            var product = (AssemblyProductAttribute)app.GetCustomAttributes(typeof(AssemblyProductAttribute), false)[0];

            return product.Product;
        }

        public static string GetCompanyName()
        {
            var app = Assembly.GetEntryAssembly();
            var company = (AssemblyCompanyAttribute)app.GetCustomAttributes(typeof(AssemblyCompanyAttribute), false)[0];

            return company.Company;
        }

        public static FileVersionInfo GetVersion()
        {
            var assembly = Assembly.GetEntryAssembly();
            return FileVersionInfo.GetVersionInfo(assembly.Location);
        }

        public static FileVersionInfo[] GetProjectDependenciesVersions()
        {
            var executingPath = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            var dependencies = GetDependenciesVersions();
            return dependencies.Where(x => Path.GetFileName(x.FileName).Contains("VI.")).ToArray();
        }

        public static int VIUsbAppId { get { return 500; } }
        public static int SysAdminUsbAppId { get { return 999; } }

        public static FileVersionInfo[] GetDependenciesVersions()
        {
            return Assembly.GetEntryAssembly()
                           .GetReferencedAssemblies()
                           .Select(x =>
                               FileVersionInfo.GetVersionInfo(
                                        Assembly.ReflectionOnlyLoad(x.FullName)
                                                .Location))
                           .ToArray();
        }
    }
}
