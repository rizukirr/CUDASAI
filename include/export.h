#ifndef CDS_EXPORT_H
#define CDS_EXPORT_H

#if defined(CDS_STATIC_DEFINE)
#define CDS_API
#else
#ifndef CDS_API
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef numc_EXPORTS
#define CDS_API __declspec(dllexport)
#else
#define CDS_API __declspec(dllimport)
#endif
#else
#if __GNUC__ >= 4
#define CDS_API __attribute__((visibility("default")))
#else
#define CDS_API
#endif
#endif
#endif
#endif

#endif /* CDS_EXPORT_H */
